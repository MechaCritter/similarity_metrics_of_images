"""Optional :mod:`torch` backend shared across pyvisim.

:mod:`torch` is an optional dependency installed by the ``nn`` extra
(``pip install "pyvisim[nn]"``). It is imported lazily through
:class:`~pyvisim.lazy_import.OptionalImport` so that pyvisim's classical
(non-neural) features never require it.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from .lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as torch_import:
    import torch


def is_tensor(obj: Any) -> TypeGuard[torch.Tensor]:
    """Return whether ``obj`` is a :class:`torch.Tensor`.

    Safe to call when torch is not installed: it short-circuits to ``False``
    instead of raising, so the classical pipeline can accept torch tensors
    without depending on torch.

    :param obj: Any object.
    :return: ``True`` if torch is installed and ``obj`` is a tensor.
    """
    return torch_import.is_available and isinstance(obj, torch.Tensor)
