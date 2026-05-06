"""Compatibility wrapper for :mod:`clipforge.storage.state`."""

import sys as _sys

from clipforge.storage import state as _impl

# TODO: Move callers to clipforge.storage.state and remove this shim.
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
