"""Compatibility wrapper for :mod:`clipforge.pipeline.render_clip`."""

import sys as _sys

from clipforge.pipeline import render_clip as _impl

# TODO: Move callers to clipforge.pipeline.render_clip and remove this shim.
if __name__ == "__main__":
    raise SystemExit(_impl.main())

_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
