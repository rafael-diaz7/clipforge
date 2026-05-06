"""Compatibility wrapper for :mod:`clipforge.media.layouts`."""

import sys as _sys

from clipforge.media import layouts as _impl

# TODO: Move callers to clipforge.media.layouts and remove this shim.
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
