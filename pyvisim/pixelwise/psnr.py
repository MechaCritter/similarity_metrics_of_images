"""Peak signal-to-noise ratio (PSNR) similarity metric."""

from __future__ import annotations

import numpy as np

from .._base_classes import SimilarityMetric
from ..features._utils import grayscale_dims
from ..lazy_import import is_tensor
from ..typing import Float64NumpyArray, ImageInput, UInt8NumpyArray
from ..utils.cython_utils import get_kernel_threads
from ..utils.image_utils import iter_images
from ._kernel._ssd_kernel import ssd_matrix

__all__ = ["PSNR"]

#: Peak value of the canonical ``uint8`` image range ``[0, 255]``.
_PEAK_SIGNAL = 255.0


def _stack_images(
    images: ImageInput,
    dims: str,
    value_range: tuple[float, float],
) -> UInt8NumpyArray:
    """
    Collect the canonical images of one input into a ``(N, H, W[, C])`` stack.

    :raises ValueError: If no image is given or the images differ in shape.
    """
    if isinstance(images, np.ndarray) or is_tensor(images):
        # A single channel-less array (e.g. a 2-D grayscale image with the
        # default "HWC") keeps working, like elsewhere in the library.
        dims = grayscale_dims(images, dims)
    image_list = list(iter_images(images, dims=dims, value_range=value_range))
    shapes = {image.shape for image in image_list}
    if len(shapes) > 1:
        raise ValueError(
            f"PSNR requires images of identical shape, got {sorted(shapes)}."
        )
    if not image_list:
        raise ValueError("Expected at least one image, got none.")
    if len(image_list) == 1:
        # A view instead of np.stack's copy; the batch is only read from.
        return image_list[0][np.newaxis]
    return np.stack(image_list)


def _mean_squared_error(
    batch1: UInt8NumpyArray, batch2: UInt8NumpyArray
) -> Float64NumpyArray:
    """
    Compute the pairwise mean squared error matrix between two image batches.

    :return: The ``(N, M)`` matrix of mean squared pixel errors.
    """
    if batch1.shape[1:] != batch2.shape[1:]:
        raise ValueError(
            f"PSNR requires images of identical shape, got {batch1.shape[1:]} "
            f"and {batch2.shape[1:]}."
        )
    flat1 = np.ascontiguousarray(batch1.reshape(len(batch1), -1))
    flat2 = np.ascontiguousarray(batch2.reshape(len(batch2), -1))
    totals = ssd_matrix(flat1, flat2, get_kernel_threads())
    error: Float64NumpyArray = totals / flat1.shape[1]
    return error


def _peak_signal_noise_ratio(
    batch1: UInt8NumpyArray, batch2: UInt8NumpyArray
) -> Float64NumpyArray:
    """
    Compute the pairwise PSNR matrix in decibels between two image batches.

    :raises ValueError: If the two batches do not share the same image shape.
    """
    error = _mean_squared_error(batch1, batch2)
    with np.errstate(divide="ignore"):
        return np.asarray(10.0 * np.log10(_PEAK_SIGNAL**2 / error), dtype=np.float64)


class PSNR(SimilarityMetric):
    """
    Peak signal-to-noise ratio between two batches of images.

    The metric is computed pairwise: for ``N`` images in the first batch and
    ``M`` images in the second batch, :meth:`similarity_score` returns an
    ``(N, M)`` matrix of PSNR values in decibels. Every compared pair must
    share the same ``(H, W[, C])`` shape.

    The squared differences are summed by a compiled OpenMP kernel. Set the
    ``PYVISIM_NUM_THREADS`` environment variable to override the team size.

    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
        integer.

    Example:

    >>> import numpy as np
    >>> from pyvisim.pixelwise import PSNR
    >>> image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    >>> PSNR().similarity_score(image, image)
    array([[inf]])
    """

    def __init__(self, batch_size: int = 16) -> None:
        super().__init__(batch_size=batch_size)

    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float64NumpyArray:
        """
        Compute the pairwise PSNR matrix between two image batches.

        :param image1: First (batch of) image(s) as ``MatLike`` (NumPy array,
            torch tensor or array-like).
        :param image2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: An ``(N, M)`` matrix of PSNR values in decibels. Identical
            pairs yield ``inf``.
        :raises ValueError: If a batch is empty or a compared pair of images
            differs in shape.
        """
        batch1 = _stack_images(image1, dims, value_range)
        batch2 = _stack_images(image2, dims, value_range)
        scores: Float64NumpyArray = np.empty(
            (len(batch1), len(batch2)), dtype=np.float64
        )
        whole_input = self._batch_size == -1
        step1 = len(batch1) if whole_input else self._batch_size
        step2 = len(batch2) if whole_input else self._batch_size
        for start1 in range(0, len(batch1), step1):
            chunk1 = batch1[start1 : start1 + step1]
            for start2 in range(0, len(batch2), step2):
                chunk2 = batch2[start2 : start2 + step2]
                scores[start1 : start1 + step1, start2 : start2 + step2] = (
                    _peak_signal_noise_ratio(chunk1, chunk2)
                )
        return scores

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size})"
