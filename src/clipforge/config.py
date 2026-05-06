"""Compatibility wrapper for :mod:`clipforge.core.config`."""

import sys as _sys

from clipforge.core import config as _impl

# TODO: Move callers to clipforge.core.config and remove this shim.
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
