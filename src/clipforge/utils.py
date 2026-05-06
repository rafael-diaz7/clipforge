"""Compatibility wrapper for :mod:`clipforge.core.utils`."""

import sys as _sys

from clipforge.core import utils as _impl

# TODO: Move callers to clipforge.core.utils and remove this shim.
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
