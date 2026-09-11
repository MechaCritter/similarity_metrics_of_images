import itertools
from collections.abc import Iterable, Iterator

import numpy as np

from .._errors import InvalidImageError
from ..lazy_import import is_tensor
from ..typing import ImageInput, MatLike, UInt8NumpyArray, _to_image_list


def grayscale_dims(image: MatLike, dims: str) -> str:
    """
    Drop the channel label from ``dims`` for a single-channel (grayscale) image.

    A grayscale image carries no channel axis, so an array with exactly one
    fewer dimension than a channel-bearing ``dims`` (e.g. a 2-D array with the
    default ``"HWC"``) is treated as single-channel and the ``"C"`` label is
    removed. This keeps the canonical ``(H, W)`` grayscale layout working with
    the channel-bearing default, matching the NumPy-only behaviour the library
    accepted before ``dims`` were introduced.

    :param image: The image whose axis count is inspected.
    :param dims: The requested axis-label string.
    :return: ``dims`` with ``"C"`` removed when ``image`` is single-channel,
        otherwise ``dims`` unchanged.
    """
    normalized = dims.upper()
    if "C" not in normalized:
        return dims
    if np.ndim(image) == len(normalized) - 1:
        return normalized.replace("C", "")
    return dims


def iter_images(
    images: ImageInput,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> Iterator[UInt8NumpyArray]:
    """
    Yield canonical per-image arrays from a single object or an iterable.

    A single (possibly batched) ``MatLike`` array/tensor is normalized and its
    images are yielded. Any other iterable is treated as a collection of
    ``MatLike`` images, each of which is normalized in turn; if its elements
    carry a batch axis (per ``dims``), every image is still yielded, so a batch
    size is handled gracefully.

    :param images: A single ``MatLike`` object or an iterable of ``MatLike`` images.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout, **default**);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: An iterator over ``uint8`` images of shape ``(H, W[, C])``.
    :raises InvalidImageError: If a string/bytes object is passed as an image.
    """
    if not (isinstance(images, (np.ndarray, Iterable)) or is_tensor(images)):
        raise InvalidImageError(
            f"Expected image array(s), but got a {type(images).__name__} object."
        )
    if isinstance(images, np.ndarray) or is_tensor(images):
        yield from _to_image_list(images, dims, value_range)
        return
    if isinstance(images, Iterable):
        for image in images:
            yield from _to_image_list(image, grayscale_dims(image, dims), value_range)
        return
    yield from _to_image_list(images, dims, value_range)


def iter_image_batches(
    images: ImageInput,
    batch_size: int,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> Iterator[list[UInt8NumpyArray]]:
    """
    Yield canonical per-image arrays in batches of at most ``batch_size``.

    The images are normalized by :func:`iter_images` and grouped lazily, so an
    iterable input is consumed one batch at a time and never held in memory as
    a whole. The last batch may be shorter than ``batch_size``; an input
    holding no image yields no batch at all.

    ``batch_size=-1`` yields the whole input as a single batch.

    :param images: A single ``MatLike`` object or an iterable of ``MatLike`` images.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. See :mod:`pyvisim.typing`.
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: An iterator over lists of at most ``batch_size`` ``uint8`` images
        of shape ``(H, W[, C])``.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive integer.
    :raises InvalidImageError: If a string/bytes object is passed as an image.
    """
    if batch_size != -1 and batch_size < 1:
        raise ValueError(
            "batch_size must be a positive integer or -1 (process the whole "
            f"input as one batch), got {batch_size}."
        )
    image_iterator = iter_images(images, dims=dims, value_range=value_range)
    if batch_size == -1:
        if batch := list(image_iterator):
            yield batch
        return
    while batch := list(itertools.islice(image_iterator, batch_size)):
        yield batch
