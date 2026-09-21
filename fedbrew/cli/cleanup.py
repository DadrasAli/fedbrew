"""Remove runtime, raw, and generated artifacts from the project tree."""

# This module intentionally supports Python 3.6 so cleanup works before the
# project environment is activated.

import argparse
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ROOT_RUNTIME_DIRS = (
    "outputs",
    os.path.join("data", "generated"),
    os.path.join("data", "raw"),
)


def clean_runtime_artifacts(root=PROJECT_ROOT, excludes=()):
    """Delete runtime artifacts, preserving any explicitly excluded paths."""

    root = os.path.abspath(root)
    excluded_paths = _resolve_excludes(root, excludes)
    removed = []

    for directory_name in ROOT_RUNTIME_DIRS:
        path = os.path.join(root, directory_name)
        removed.extend(_remove_path(path, excluded_paths))

    pycache_dirs = []
    pyc_files = []
    for current_root, dirs, files in os.walk(root):
        if _is_inside_git(root, current_root) or _is_excluded(current_root, excluded_paths):
            dirs[:] = []
            continue
        dirs[:] = [
            dirname
            for dirname in dirs
            if not _is_excluded(os.path.join(current_root, dirname), excluded_paths)
        ]
        for dirname in dirs:
            if dirname == "__pycache__":
                pycache_dirs.append(os.path.join(current_root, dirname))
        for filename in files:
            if filename.endswith(".pyc"):
                pyc_files.append(os.path.join(current_root, filename))

    for path in sorted(pycache_dirs, key=_sort_key):
        removed.extend(_remove_path(path, excluded_paths))

    for path in sorted(pyc_files, key=_sort_key):
        if _inside_removed_directory(path, removed):
            continue
        removed.extend(_remove_path(path, excluded_paths))

    return removed


def _resolve_excludes(root, excludes):
    """Resolve exclusions relative to *root* and keep them within the project."""

    resolved = []
    for value in excludes:
        candidate = os.path.expanduser(value)
        if not os.path.isabs(candidate):
            candidate = os.path.join(root, candidate)
        candidate = os.path.abspath(candidate)
        try:
            inside_root = os.path.commonpath((root, candidate)) == root
        except ValueError as exc:
            raise ValueError(f"excluded path is outside the project: {value}") from exc
        if not inside_root:
            raise ValueError(f"excluded path is outside the project: {value}")
        resolved.append(candidate)
    return tuple(sorted(set(resolved), key=_sort_key))


def _remove_path(path, excluded_paths):
    """Remove one path, descending only when it contains an exclusion."""

    if _is_excluded(path, excluded_paths):
        return []
    if not os.path.exists(path) and not os.path.islink(path):
        return []
    if not os.path.isdir(path) or os.path.islink(path):
        os.unlink(path)
        return [path]
    if not _contains_exclusion(path, excluded_paths):
        shutil.rmtree(path)
        return [path]

    removed = []
    for name in os.listdir(path):
        removed.extend(_remove_path(os.path.join(path, name), excluded_paths))
    if not os.listdir(path):
        os.rmdir(path)
        removed.append(path)
    return removed


def _is_excluded(path, excluded_paths):
    return os.path.abspath(path) in excluded_paths


def _contains_exclusion(path, excluded_paths):
    path = os.path.abspath(path)
    prefix = path + os.sep
    return any(excluded == path or excluded.startswith(prefix) for excluded in excluded_paths)


def _inside_removed_directory(path, removed):
    path = os.path.abspath(path)
    return any(
        path == os.path.abspath(parent) or path.startswith(os.path.abspath(parent) + os.sep)
        for parent in removed
    )


def _is_inside_git(root, path):
    relative = os.path.relpath(path, root)
    return relative == ".git" or relative.startswith(".git" + os.sep)


def _sort_key(path):
    return (-len(os.path.abspath(path).split(os.sep)), path)


def parse_args(argv=None):
    """Parse cleanup command arguments."""

    parser = argparse.ArgumentParser(
        prog="fedbrew cleanup",
        description="Remove runtime artifacts while preserving selected paths.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="Keep a project-relative file or directory. Can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run cleanup and print removed paths."""

    args = parse_args(argv)
    try:
        excluded_paths = _resolve_excludes(PROJECT_ROOT, args.exclude)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if excluded_paths:
        print("Protected paths:")
        for path in excluded_paths:
            print(f"- {os.path.relpath(path, PROJECT_ROOT)}")

    removed = clean_runtime_artifacts(PROJECT_ROOT, excludes=excluded_paths)
    if not removed:
        print("No runtime artifacts found.")
        return

    print("Removed runtime artifacts:")
    for path in removed:
        print(f"- {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
