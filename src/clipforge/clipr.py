"""Compatibility wrapper for :mod:`clipforge.integrations.clipr`."""

import sys as _sys

from clipforge.integrations import clipr as _impl

# TODO: Move callers to clipforge.integrations.clipr and remove this shim.
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
