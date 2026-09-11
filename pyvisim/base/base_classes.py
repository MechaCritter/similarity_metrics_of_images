import abc
from collections.abc import Iterator

import numpy as np

from .._base_classes import SimilarityMetric
from ..lazy_import import is_tensor
from ..typing import Float64NumpyArray, FloatNumpyArray, ImageInput, IntNumpyArray
from ..utils.image_utils import grayscale_dims, iter_images


def _stack_image_batch(
    images: ImageInput,
    dims: str,
    value_range: tuple[float, float],
) -> Float64NumpyArray:
    """
    Normalize ``images`` and stack them into one ``(N, H, W, C)`` float batch.

    Every image is first converted to the canonical ``uint8`` ``(H, W[, C])``
    layout in ``[0, 255]`` (see :mod:`pyvisim.typing`), then the batch is
    stacked and cast to ``float64`` for numerically stable arithmetic.
    Grayscale images receive a singleton channel axis.

    :param images: A single ``MatLike`` image, a batched array, or an iterable
        of images.
    :param dims: Axis-label string describing the input axes (see
        :mod:`pyvisim.typing`).
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A ``(N, H, W, C)`` ``float64`` array with values in ``[0, 255]``.
    :raises InvalidImageError: If an input cannot be converted to a numeric
        array.
    :raises ValueError: If no image is given or the images differ in shape.
    """
    if isinstance(images, np.ndarray) or is_tensor(images):
        # A single channel-less array (e.g. a 2-D grayscale image with the
        # default "HWC") keeps working, like elsewhere in the library.
        dims = grayscale_dims(images, dims)
    canonical = list(iter_images(images, dims=dims, value_range=value_range))
    if not canonical:
        raise ValueError("Expected at least one image, got none.")
    shapes = {image.shape for image in canonical}
    if len(shapes) > 1:
        raise ValueError(
            "All images in a batch must have the same shape to be compared "
            f"pixel-wise, got shapes {sorted(shapes)}."
        )
    batch = np.stack(canonical).astype(np.float64)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    return batch


def _iter_pair_chunks(
    n_rows: int, n_cols: int, batch_size: int
) -> Iterator[tuple[IntNumpyArray, IntNumpyArray]]:
    """
    Yield ``(rows, cols)`` index arrays covering an ``n_rows x n_cols`` grid.

    Pairs are enumerated in row-major order and grouped into chunks of at most
    ``batch_size`` pairs; ``-1`` yields every pair in a single chunk.

    :param n_rows: Number of images in the first batch.
    :param n_cols: Number of images in the second batch.
    :param batch_size: Maximum number of image pairs processed in a single
        batch. Set to ``-1`` to process all images as a single batch.
    :return: An iterator of ``(rows, cols)`` integer index arrays.
    """
    n_pairs = n_rows * n_cols
    chunk = n_pairs if batch_size == -1 else batch_size
    for start in range(0, n_pairs, chunk):
        flat = np.arange(start, min(start + chunk, n_pairs), dtype=np.intp)
        yield flat // n_cols, flat % n_cols


class DenseMetricBase(SimilarityMetric, abc.ABC):
    """
    Base class for metrics that score two aligned pixel grids directly.

    Concrete subclasses implement ``_score_pairs``, which receives two stacked
    ``float64`` batches of identical shape and returns one score per pair.

    :param batch_size: Maximum number of image pairs processed in a single
        batch. Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor positive.
    """

    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute the pairwise score matrix between two image batches.

        Every image is normalized to the canonical ``uint8`` ``(H, W[, C])``
        layout in ``[0, 255]`` first, so the metric always operates on the
        same value scale regardless of the input dtype or range. The pairs are
        scored in chunks of at most :attr:`batch_size` pairs.

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
        :return: A ``(N, M)`` matrix scoring every image of ``image1`` against
            every image of ``image2``.
        :raises InvalidImageError: If an input cannot be converted to a numeric
            array.
        :raises ValueError: If a batch is empty, the two batches hold images of
            different shapes, or the images are too small for the metric.
        """
        batch1 = _stack_image_batch(image1, dims, value_range)
        batch2 = _stack_image_batch(image2, dims, value_range)
        if batch1.shape[1:] != batch2.shape[1:]:
            raise ValueError(
                "image1 and image2 must contain images of the same shape, "
                f"got {batch1.shape[1:]} vs {batch2.shape[1:]}."
            )
        self._validate_image_shape(batch1.shape[1], batch1.shape[2])
        scores = np.empty((batch1.shape[0], batch2.shape[0]), dtype=np.float64)
        for rows, cols in _iter_pair_chunks(
            batch1.shape[0], batch2.shape[0], self._batch_size
        ):
            scores[rows, cols] = self._score_pairs(batch1[rows], batch2[cols])
        return scores

    def _validate_image_shape(self, height: int, width: int) -> None:
        """
        Hook for subclasses to reject images too small for the metric.

        The default accepts any size.

        :param height: Height of the images, in pixels.
        :param width: Width of the images, in pixels.
        :raises ValueError: If the images cannot be scored by this metric.
        """

    @abc.abstractmethod
    def _score_pairs(
        self, images1: Float64NumpyArray, images2: Float64NumpyArray
    ) -> Float64NumpyArray:
        """
        Score aligned image pairs.

        :param images1: ``(B, H, W, C)`` ``float64`` batch, one image per pair.
        :param images2: ``(B, H, W, C)`` ``float64`` batch, aligned with
            ``images1``.
        :return: A ``(B,)`` array holding one score per pair.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size})"
