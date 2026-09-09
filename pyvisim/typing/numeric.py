"""
Typing utilities and image-normalization helpers for pyvisim.

This module defines :data:`MatLike`, the permissive image type accepted across
the public API, and the helpers that turn any ``MatLike`` input into the
canonical NumPy layout the rest of the library works with.

Canonical layout
=================
Every image is normalized to a ``uint8`` NumPy array of shape ``(H, W, C)``
(or ``(H, W)`` for single-channel images) with values in the ``[0, 255]``
range. SIFT/RootSIFT and :class:`DeepConvFeature` (torchvision
transforms) both expect this representation, so normalizing once keeps the
downstream code unified.

Dimension labels (``dims``)
===========================
The ``dims`` string tells the helpers how to read the axes of an input array.
Each character names one axis, in the exact order the axes appear:

- ``"B"``: batch axis (number of images).
- ``"H"``: height axis (number of rows).
- ``"W"``: width axis (number of columns).
- ``"C"``: channel axis (e.g. RGB color channels).

Common examples:

- ``"HWC"``: a single image laid out height x width x channels
  (the classic NumPy/OpenCV layout). This is the default.
- ``"CHW"``: a single image laid out channels x height x width
  (the classic PyTorch layout).
- ``"BHWC"``: a batch of images, batch x height x width x channels.
- ``"BCHW"``: a batch of images, batch x channels x height x width
  (the PyTorch batched layout).
- ``"HWCB"``: height x width x channels x batch.

``"B"`` and ``"C"`` are optional; ``"H"`` and ``"W"`` are mandatory. When the
``dims`` string contains a batch axis, the input is split into the individual
images it contains.

Value range (``value_range``)
=============================
``value_range`` is the ``(low, high)`` range the *input* values live in. It
defaults to ``(0.0, 255.0)`` (standard 8-bit images). If a different range is
provided, e.g. ``(0.0, 1.0)`` for normalized float tensors, the values are
rescaled into the canonical ``[0, 255]`` range.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from .._errors import InvalidImageError
from ..lazy_import import is_tensor

#: Generic NumPy array of any dtype.
NumpyArray = npt.NDArray[np.generic]
#: NumPy array of uint8 values (canonical image layout).
UInt8NumpyArray = npt.NDArray[np.uint8]
#: NumPy array of float32 values (feature descriptors and embeddings).
Float32NumpyArray = npt.NDArray[np.float32]
#: NumPy array of float64 values (model outputs computed in double precision).
Float64NumpyArray = npt.NDArray[np.float64]
#: NumPy array of any floating-point dtype.
FloatNumpyArray = npt.NDArray[np.floating[Any]]
#: NumPy array of platform-native signed integers (cluster labels / indices).
IntNumpyArray = npt.NDArray[np.intp]
#: Boolean matrix.
BoolNumpyArray = npt.NDArray[np.bool_]

#: A similarity function: maps two batches of feature vectors of shapes
#: ``(N, D)`` and ``(M, D)`` to an ``(N, M)`` similarity matrix.
SimilarityFunc = Callable[[FloatNumpyArray, FloatNumpyArray], FloatNumpyArray]

#: Anything that can be turned into a numerical NumPy array: a NumPy array, a
#: PyTorch tensor, or any array-like object (e.g. nested lists of numbers).
#: ``torch.Tensor`` is part of the union only when torch (the ``nn`` extra) is
#: installed; without torch the alias collapses to ``npt.ArrayLike`` at runtime.
if TYPE_CHECKING:
    import torch

    MatLike = torch.Tensor | npt.ArrayLike
else:
    MatLike = npt.ArrayLike

#: A single image or a collection of images. Either one ``MatLike`` object
#: (optionally carrying a batch axis) or an iterable of ``MatLike`` images.
ImageInput = MatLike | Iterable[MatLike]

_VALID_DIM_CHARS = frozenset("BHWC")
_CANONICAL_AXIS_ORDER = "BHWC"
_DEFAULT_VALUE_RANGE: tuple[float, float] = (0.0, 255.0)


def _to_ndarray(data: MatLike) -> NumpyArray:
    """
    Convert any ``MatLike`` object into a numerical NumPy array.

    :param data: A NumPy array, a PyTorch tensor, or any array-like object.
    :return: The data as a NumPy array.
    :raises InvalidImageError: If the data cannot be turned into a numeric array.
    """
    if is_tensor(data):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        array = data
    else:
        try:
            array = np.asarray(data)
        except (ValueError, TypeError) as exc:
            raise InvalidImageError(
                f"Could not convert object of type {type(data).__name__!r} "
                "to a NumPy array."
            ) from exc
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.bool_
    ):
        raise InvalidImageError(
            f"Expected a numeric array, but got an array with dtype {array.dtype!r}."
        )
    return array


def _validate_dims(dims: str, ndim: int) -> str:
    """
    Validate a ``dims`` string against the number of array dimensions.

    :param dims: Axis-label string, one character per axis (case-insensitive).
        ``"H"`` = height, ``"W"`` = width, ``"C"`` = channels, ``"B"`` = batch.
        For example, ``"HWC"`` is height × width × channels (the default).
    :param ndim: Number of dimensions of the array the labels describe.
    :return: The normalized (upper-cased) ``dims`` string.
    :raises ValueError: If the string is malformed or inconsistent with ``ndim``.
    """
    normalized = dims.upper()
    if len(normalized) != ndim:
        raise ValueError(
            f"'dims' string {dims!r} describes {len(normalized)} axis/axes, "
            f"but the array has {ndim} dimension(s)."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"'dims' string {dims!r} contains duplicate axis labels.")
    invalid = set(normalized) - _VALID_DIM_CHARS
    if invalid:
        raise ValueError(
            f"'dims' string {dims!r} contains invalid axis labels {sorted(invalid)}. "
            "Only 'B' (batch), 'H' (height), 'W' (width) and 'C' (channels) are allowed."
        )
    if "H" not in normalized or "W" not in normalized:
        raise ValueError(
            f"'dims' string {dims!r} must contain both 'H' (height) and 'W' (width)."
        )
    return normalized


def _to_uint8(array: NumpyArray, value_range: tuple[float, float]) -> UInt8NumpyArray:
    """
    Rescale an array from ``value_range`` into the canonical ``[0, 255]`` uint8 range.

    :param array: Numeric array to rescale.
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A ``uint8`` array with values clipped to ``[0, 255]``.
    :raises ValueError: If ``value_range`` is not strictly increasing.
    """
    low, high = value_range
    if high <= low:
        raise ValueError(
            f"'value_range' must be an increasing (low, high) tuple, got {value_range}."
        )
    if value_range == _DEFAULT_VALUE_RANGE and array.dtype == np.uint8:
        return cast(UInt8NumpyArray, array)
    scaled = (array.astype(np.float64) - low) / (high - low) * 255.0
    scaled = np.clip(scaled, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _split_into_images(array: UInt8NumpyArray, dims: str) -> list[UInt8NumpyArray]:
    """
    Reorder an array into ``(B, H, W[, C])`` and split it along the batch axis.

    :param array: Array whose axes are described by ``dims``.
    :param dims: Validated, normalized ``dims`` string.
    :return: A list of contiguous per-image arrays of shape ``(H, W[, C])``.
    """
    target = "".join(axis for axis in _CANONICAL_AXIS_ORDER if axis in dims)
    reordered = np.transpose(array, [dims.index(axis) for axis in target])
    if "B" not in dims:
        reordered = reordered[np.newaxis, ...]
    return [
        np.ascontiguousarray(reordered[index]) for index in range(reordered.shape[0])
    ]


def _to_image_list(
    images: MatLike,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> list[UInt8NumpyArray]:
    """
    Normalize a single ``MatLike`` object into canonical per-image arrays.

    The input may carry a batch axis (when ``dims`` contains ``"B"``), in which
    case it is split into the individual images it holds. See the module
    docstring for the meaning of ``dims`` and ``value_range``.

    :param images: A NumPy array, a PyTorch tensor, or any array-like object.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout, **default**);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: A list of ``uint8`` images of shape ``(H, W[, C])`` in ``[0, 255]``.
    :raises InvalidImageError: If the input cannot be converted to a numeric array.
    :raises ValueError: If ``dims`` or ``value_range`` are invalid.
    """
    array = _to_ndarray(images)
    normalized_dims = _validate_dims(dims, array.ndim)
    array = _to_uint8(array, value_range)
    return _split_into_images(array, normalized_dims)
