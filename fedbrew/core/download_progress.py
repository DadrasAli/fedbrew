"""Redirect third-party download progress into fedbrew's own console output.

huggingface_hub, datasets and torchvision each drive their download progress
bars through a tqdm-compatible class bound as a plain attribute on one of
their own modules; none exposes a callback parameter for it. Confirmed by
reading each library's source against the versions this env has installed
(huggingface_hub 0.36.2, datasets 2.21.0, torchvision 0.15.2a0):

- ``huggingface_hub/utils/tqdm.py`` defines a ``tqdm`` class; ``file_download.py``
  (which ``hf_hub_download``/``cached_file`` -- and so ``transformers``'
  ``from_pretrained`` -- use for the actual byte-level download) resolves it
  by bare name from that same module's globals at call time, inside
  ``_get_progress_bar_context``.
- ``datasets/utils/tqdm.py`` defines an equivalent ``tqdm`` class, resolved
  the same way by both ``http_get`` and ``fsspec_get`` in
  ``datasets/utils/file_utils.py`` -- both of ``load_dataset``'s download
  paths.
- ``torchvision/datasets/utils.py`` does ``from torch.utils.model_zoo import
  tqdm`` at module scope and calls it directly in ``_save_response_content``.

transformers needs no entry here: ``transformers/utils/hub.py`` calls
huggingface_hub's own ``hf_hub_download``/``snapshot_download`` directly and
has no download-progress mechanism of its own, so redirecting
huggingface_hub's attribute covers it.

Not covered: ``huggingface_hub.utils.snapshot_download`` binds its own
default ``tqdm`` class at import time (``from .utils import tqdm as
hf_tqdm``), which this redirect -- patching the attribute afterward -- cannot
reach. That path is the "Fetching N files" bar for a multi-file snapshot
download; every model this repo ships a config for loads through
``cached_file``/``hf_hub_download`` for a single (unsharded) weights file
instead, so it is not exercised in practice. Chasing it would also mean a
second, outer progress line for the same download -- exactly the nesting
this was asked not to build.

All three libraries are optional here -- huggingface_hub and datasets ship
only in the ``llm`` extra, torchvision only in ``vision`` -- so an install
without one is a normal state rather than an error. Absence therefore
degrades to a no-op context manager, exactly as a moved attribute degrades
to the library's own bars: the callers that actually need the library
(``prepare.py``, ``oasst1.py``, ``generate.py``) each raise their own
message naming the extra to install, and a progress helper must not
pre-empt that with a ModuleNotFoundError of its own.

Swapping the attribute for a subclass that reports through a caller-supplied
callback instead of rendering is the same integration point tqdm's own
ecosystem (``tqdm.contrib.slack``/``discord``/``telegram``) uses to redirect
output elsewhere -- not blind suppression: bytes-received and total genuinely
flow through ``update()``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

#: (name, done_bytes, total_bytes_or_None, elapsed_seconds, finished=...).
#: Called throttled to roughly 10/second while a download is in progress,
#: plus always once with finished=True when it completes.
ProgressCallback = Callable[..., None]

_MIN_REPORT_INTERVAL_SECONDS = 0.1


def _driven_tqdm_class(base: type, on_progress: ProgressCallback) -> type:
    """Build a tqdm-compatible subclass of `base` that reports through
    `on_progress` instead of rendering.

    Cannot read anything back from tqdm.__init__ after forcing disable=True:
    tqdm.std.tqdm.__init__ takes an early-return path when disabled that
    never sets self.desc (checked against tqdm 4.67.3's std.py), and
    update()/close() return immediately without touching self.n when
    disabled. So everything this class needs is captured from the
    constructor arguments before delegating, and progress is tracked
    independently rather than through the base class's own counters.
    """

    class _DrivenTqdm(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._fedbrew_name = str(
                kwargs.get("desc") or (args[0] if args else None) or "download"
            )
            self._fedbrew_total = kwargs.get("total")
            self._fedbrew_done = kwargs.get("initial") or 0
            self._fedbrew_started = time.perf_counter()
            self._fedbrew_last_reported = 0.0
            self._fedbrew_closed = False
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._fedbrew_report(finished=False)

        def update(self, n: int = 1) -> None:
            self._fedbrew_done += n
            self._fedbrew_report(finished=False)

        def close(self) -> None:
            # tqdm's own close() is idempotent (it disables itself the first
            # time), and callers routinely close a bar more than once: a
            # `with tqdm(...)` block's __exit__ plus tqdm's own __del__
            # safety net both call it, and both fired here in practice
            # against a real torchvision download before this guard existed
            # -- one file reported "finished" twice.
            if self._fedbrew_closed:
                return
            self._fedbrew_closed = True
            self._fedbrew_report(finished=True)

        def _fedbrew_report(self, *, finished: bool) -> None:
            now = time.perf_counter()
            if not finished and now - self._fedbrew_last_reported < _MIN_REPORT_INTERVAL_SECONDS:
                return
            self._fedbrew_last_reported = now
            try:
                on_progress(
                    self._fedbrew_name,
                    self._fedbrew_done,
                    self._fedbrew_total,
                    now - self._fedbrew_started,
                    finished=finished,
                )
            except Exception:
                # A bug in our own reporting must never look like the
                # download itself failed -- callers of the redirected
                # functions (generate.py's MNIST loader in particular) treat
                # most exceptions from the wrapped call as "torchvision
                # failed, fall back" and would silently mis-attribute one
                # here.
                pass

    _DrivenTqdm.__name__ = f"Driven{base.__name__}"
    return _DrivenTqdm


@contextmanager
def _redirect_progress(
    module: Any,
    attr: str,
    on_progress: ProgressCallback,
) -> Iterator[bool]:
    """Replace `module.<attr>` with a driven subclass for the block's
    duration, and restore the original afterward -- on a normal return and
    on an exception alike.

    Yields whether the redirect actually took effect. False means either
    that `module` is None -- the library is not installed at all, see
    `_submodule` -- or that `attr` was not found on `module`, or was not a
    class: the library moved or renamed it since this was written.
    Degrading in both cases -- to nothing at all when the library is
    absent, to the library's own progress bars when it is present but has
    moved the attribute -- rather than raising, or silently patching
    nothing while claiming success, is deliberate: a version bump or a
    missing optional extra should cost the redirect, not the download.
    """

    if module is None:
        yield False
        return

    original = getattr(module, attr, None)
    if not isinstance(original, type):
        yield False
        return

    setattr(module, attr, _driven_tqdm_class(original, on_progress))
    try:
        yield True
    finally:
        setattr(module, attr, original)


def redirect_huggingface_hub_progress(on_progress: ProgressCallback) -> Any:
    """Context manager: drive `on_progress` from huggingface_hub's download
    progress for its duration (model/tokenizer downloads via
    ``from_pretrained``, and so transformers' too)."""

    return _redirect_progress(_submodule("huggingface_hub.utils.tqdm"), "tqdm", on_progress)


def redirect_datasets_progress(on_progress: ProgressCallback) -> Any:
    """Context manager: drive `on_progress` from `datasets`' download
    progress for its duration (``load_dataset``)."""

    return _redirect_progress(_submodule("datasets.utils.tqdm"), "tqdm", on_progress)


def redirect_torchvision_progress(on_progress: ProgressCallback) -> Any:
    """Context manager: drive `on_progress` from torchvision's download
    progress for its duration (``MNIST``/``CIFAR10`` with ``download=True``)."""

    return _redirect_progress(_submodule("torchvision.datasets.utils"), "tqdm", on_progress)


def _submodule(dotted_name: str) -> Any:
    """Import and return the real module object at `dotted_name`, or None if
    it cannot be imported.

    Not `import a.b.c; a.b.c` -- confirmed against the installed
    huggingface_hub 0.36.2 and datasets 2.21.0, both `utils/__init__.py`
    files do ``from . import tqdm as _tqdm`` (their own comment: "_tqdm is
    the module") specifically because they *also* do ``from .tqdm import
    ..., tqdm, ...`` a few lines later, which rebinds the package's own
    ``tqdm`` attribute to the class and shadows the submodule there. Plain
    attribute access after that point returns the class, not the module --
    this is exactly the bug a first version of this file had, caught by
    RedirectFactoryTests trying to read `.tqdm` off the class and getting
    `AttributeError: type object 'tqdm' has no attribute 'tqdm'`.
    ``importlib.import_module`` sidesteps it entirely: it returns what
    Python's own module registry holds for that dotted path, unaffected by
    what any package `__init__.py` chose to re-export under a colliding
    name.
    """

    import importlib

    try:
        return importlib.import_module(dotted_name)
    except ImportError:
        # ImportError rather than only ModuleNotFoundError: a half-installed
        # or otherwise broken library fails the import the same way, and in
        # both cases the honest error belongs to the download call the
        # caller is about to make -- which raises its own message naming the
        # extra -- not to setting up that call's progress bar. The guard
        # tests keep this from hiding a *present* library whose attribute
        # moved: with the library importable they assert the redirect goes
        # active, so a swallowed import there fails the suite.
        return None
