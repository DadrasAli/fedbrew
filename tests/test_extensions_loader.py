"""An out-of-tree component reaches the registries through one loader.

``register_builtin_components`` was the only caller of ``Registry.register``
that the CLI reached, so a task, a model or a generator defined outside the
package could only be registered by whoever started the process -- which is
why every example shipped a ``run.py`` that imported its ``problem.py`` and
called the runner directly, past ``--validate-only``, the plan header and
every load-time guard.

The loader takes the entries a config writes -- a path to a ``.py`` file or
a dotted module name -- imports each, and calls its ``register()`` once. What
it checks: that both forms load, that a location is loaded once per process,
that the names an extension registers carry the entry as their origin, that a
module without ``register()`` is refused with the protocol in the message,
and that the record carries what run.json will need to say which version of
the file ran.

Registrations here go into the process-wide registries, as they would in a
run, and are popped again on cleanup so the documentation guards -- which
diff the built-in set -- see nothing.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest

from fedbrew.core import extensions, registry
from fedbrew.core.extensions import LoadedExtension, load_extensions, loaded_extensions
from fedbrew.core.refusal import RunRefused

pytestmark = pytest.mark.fast

EXTENSION = textwrap.dedent(
    '''
    """A minimal extension: one task, registered from register()."""

    from fedbrew.core import registry

    IMPORTED = []


    def register():
        registry.tasks.register("{name}", lambda **kwargs: None)
    '''
)


def _write(directory: Path, name: str, task_name: str, body: str = EXTENSION) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body.format(name=task_name), encoding="utf-8")
    return path


class LoaderFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        # The once-per-location cache is process-wide, as it must be; each
        # test starts from an empty one and leaves it empty.
        self._saved = dict(extensions._loaded)
        extensions._loaded.clear()
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        extensions._loaded.clear()
        extensions._loaded.update(self._saved)

    def _forget(self, *task_names: str) -> None:
        for name in task_names:
            self.addCleanup(registry.tasks._items.pop, name, None)
            self.addCleanup(registry.tasks._origins.pop, name, None)


class FilePathTest(LoaderFixture):
    def test_a_file_is_imported_and_its_register_called_once(self) -> None:
        path = _write(self.directory, "problem.py", "loader_probe_file")
        self._forget("loader_probe_file")

        records = load_extensions([str(path)])

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, LoadedExtension)
        self.assertEqual(record.entry, str(path))
        self.assertEqual(record.resolved, str(path.resolve()))
        self.assertEqual(record.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(record.registered, (("tasks", "loader_probe_file"),))
        self.assertTrue(registry.tasks.exists("loader_probe_file"))
        self.assertEqual(registry.tasks.origin("loader_probe_file"), str(path))

    def test_a_relative_path_is_resolved_against_the_working_directory(self) -> None:
        """The same rule as data.path, so a config that names both names them alike."""

        _write(self.directory, "problem.py", "loader_probe_relative")
        self._forget("loader_probe_relative")

        # chdir rather than a patch: resolving a relative path reads the
        # process's working directory, and changing it is what a user does.
        previous = Path.cwd()
        os.chdir(self.directory)
        try:
            records = load_extensions(["problem.py"])
        finally:
            os.chdir(previous)

        self.assertEqual(records[0].entry, "problem.py")
        self.assertEqual(records[0].resolved, str((self.directory / "problem.py").resolve()))
        self.assertEqual(registry.tasks.origin("loader_probe_relative"), "problem.py")

    def test_a_second_load_of_the_same_file_returns_the_record_without_registering(self) -> None:
        path = _write(self.directory, "problem.py", "loader_probe_twice")
        self._forget("loader_probe_twice")

        first = load_extensions([str(path)])[0]
        second = load_extensions([str(path), str(path)])

        self.assertIs(second[0], first)
        self.assertIs(second[1], first)
        self.assertEqual(loaded_extensions(), [first])

    def test_a_missing_file_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_extensions([str(self.directory / "absent.py")])
        message = str(caught.exception)
        self.assertIn("absent.py", message)
        self.assertIn("does not exist", message)

    def test_a_module_without_register_is_refused_with_the_protocol(self) -> None:
        path = self.directory / "no_hook.py"
        path.write_text("VALUE = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            load_extensions([str(path)])
        message = str(caught.exception)
        self.assertIn("no register() function", message)
        self.assertIn("nothing is registered at import", message)
        self.assertEqual(loaded_extensions(), [], "a refused module is not recorded as loaded")

    def test_a_register_that_is_not_callable_is_refused(self) -> None:
        path = self.directory / "not_callable.py"
        path.write_text("register = 3\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_extensions([str(path)])

    def test_two_files_with_the_same_basename_do_not_shadow_each_other(self) -> None:
        first_dir = self.directory / "a"
        second_dir = self.directory / "b"
        first_dir.mkdir()
        second_dir.mkdir()
        first = _write(first_dir, "problem.py", "loader_probe_a")
        second = _write(second_dir, "problem.py", "loader_probe_b")
        self._forget("loader_probe_a", "loader_probe_b")

        load_extensions([str(first), str(second)])

        self.assertTrue(registry.tasks.exists("loader_probe_a"))
        self.assertTrue(registry.tasks.exists("loader_probe_b"))
        self.assertNotIn("problem", sys.modules, "the file was imported under its bare name")

    def test_a_name_collision_between_two_files_names_both(self) -> None:
        first = _write(self.directory / "a", "problem.py", "loader_probe_clash")
        second = _write(self.directory / "b", "problem.py", "loader_probe_clash")
        self._forget("loader_probe_clash")

        with self.assertRaises(ValueError) as caught:
            load_extensions([str(first), str(second)])
        message = str(caught.exception)
        self.assertIn(str(first), message)
        self.assertIn(str(second), message)

    def test_an_import_error_inside_the_file_propagates_with_its_own_message(self) -> None:
        path = self.directory / "broken.py"
        path.write_text("import module_that_does_not_exist_anywhere\n", encoding="utf-8")
        with self.assertRaises(ModuleNotFoundError):
            load_extensions([str(path)])
        self.assertEqual(loaded_extensions(), [])


class ModuleNameTest(LoaderFixture):
    def test_a_dotted_name_is_imported_and_registered(self) -> None:
        package = self.directory / "loader_probe_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        _write(package, "problem.py", "loader_probe_module")
        self._forget("loader_probe_module")
        sys.path.insert(0, str(self.directory))
        self.addCleanup(sys.path.remove, str(self.directory))
        self.addCleanup(sys.modules.pop, "loader_probe_pkg.problem", None)
        self.addCleanup(sys.modules.pop, "loader_probe_pkg", None)

        record = load_extensions(["loader_probe_pkg.problem"])[0]

        self.assertEqual(record.entry, "loader_probe_pkg.problem")
        self.assertEqual(record.resolved, "loader_probe_pkg.problem")
        self.assertEqual(
            record.sha256,
            hashlib.sha256((package / "problem.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(registry.tasks.origin("loader_probe_module"), "loader_probe_pkg.problem")

    def test_a_module_name_that_does_not_exist_is_refused_by_name(self) -> None:
        """The entry, or a package it sits in, is not there: the config named nothing."""

        for entry, missing in (
            ("no_such_package_anywhere.problem", "no_such_package_anywhere"),
            ("no_such_module_anywhere", "no_such_module_anywhere"),
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(RunRefused) as caught:
                    load_extensions([entry])
                self.assertIn(f"no module named {missing!r}", str(caught.exception))

    def test_an_import_error_inside_a_named_module_keeps_its_traceback(self) -> None:
        """The module exists and its own import fails: the extension's defect, not the config's."""

        (self.directory / "loader_probe_broken_dependency.py").write_text(
            "import module_that_does_not_exist_anywhere\n", encoding="utf-8"
        )
        sys.path.insert(0, str(self.directory))
        self.addCleanup(sys.path.remove, str(self.directory))
        self.addCleanup(sys.modules.pop, "loader_probe_broken_dependency", None)

        with self.assertRaises(ModuleNotFoundError) as caught:
            load_extensions(["loader_probe_broken_dependency"])
        self.assertEqual(caught.exception.name, "module_that_does_not_exist_anywhere")
        self.assertEqual(loaded_extensions(), [])

    def test_an_entry_that_is_neither_form_is_refused(self) -> None:
        # "problem" and "problem.txt" are not here: both are well-formed
        # module names, and the loader tries to import them.
        for entry in ("some/dir/problem", "a-b.c", "-x", "problem file"):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError) as caught:
                    load_extensions([entry])
                self.assertIn(
                    "neither a path ending in .py nor a dotted module name",
                    str(caught.exception),
                )


class EntriesShapeTest(LoaderFixture):
    def test_a_bare_string_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            load_extensions("problem.py")  # type: ignore[arg-type]

    def test_a_non_string_entry_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            load_extensions([3])  # type: ignore[list-item]

    def test_an_empty_list_loads_nothing(self) -> None:
        self.assertEqual(load_extensions([]), [])
        self.assertEqual(loaded_extensions(), [])


if __name__ == "__main__":
    unittest.main()
