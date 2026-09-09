"""
Single-scale Structural Similarity (SSIM) metric.

References
==========
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
quality assessment: from error visibility to structural similarity. IEEE
Transactions on Image Processing, 13(4), 600-612.
https://doi.org/10.1109/TIP.2003.819861
"""

import numpy as np

from ..base import CANONICAL_DATA_RANGE, DenseMetricBase
from ..typing import Float64NumpyArray
from ..utils.cython_utils import get_kernel_threads
from ._filters import _as_planes, gaussian_kernel
from ._kernel._ssim_kernels import ssim_plane_sums

__all__ = ["SSIM"]


def _validate_window(window_size: int, sigma: float) -> None:
    """
    Validate the Gaussian window configuration.

    :param window_size: Side length of the window, in pixels.
    :param sigma: Standard deviation of the Gaussian window.
    :raises ValueError: If ``window_size`` is not an odd integer >= 3 or
        ``sigma`` is not positive.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")


def _validate_stabilizers(k1: float, k2: float) -> None:
    """
    Validate the SSIM stabilization constants.

    :param k1: Luminance stabilization constant.
    :param k2: Contrast stabilization constant.
    :raises ValueError: If either constant is not positive.
    """
    if k1 <= 0 or k2 <= 0:
        raise ValueError(f"k1 and k2 must be > 0, got k1={k1} and k2={k2}.")


class SSIM(DenseMetricBase):
    """
    Structural Similarity index (Wang et al., 2004) between image batches.

    SSIM compares two aligned images through local luminance, contrast and
    structure statistics estimated under a sliding Gaussian window::

        SSIM(x, y) = (2*mu_x*mu_y + c1) * (2*cov_xy + c2)
                     -------------------------------------------
                     (mu_x^2 + mu_y^2 + c1) * (var_x + var_y + c2)

    where ``c1 = (k1 * L) ** 2`` and ``c2 = (k2 * L) ** 2``. Windows are
    applied in "valid" mode with population statistics, and the final score is
    the mean over the whole SSIM map (all channels pooled), matching the
    author's reference implementation. Because every input is normalized to
    the canonical ``[0, 255]`` range, the dynamic range ``L`` is fixed at 255.

    Higher values mean more similar: identical images score 1, structurally
    unrelated images score near 0 and inverted structures can go negative.

    :param window_size: Side length of the Gaussian window, an odd integer
        >= 3 (default 11).
    :param sigma: Standard deviation of the Gaussian window (default 1.5).
    :param k1: Luminance stabilization constant (default 0.01).
    :param k2: Contrast stabilization constant (default 0.03).
    :param batch_size: Maximum number of image pairs processed in a single
        batch. Set to ``-1`` to process all images as a single batch.
    :param num_workers: Number of threads to use for the computation. If
        ``None``, use the default ``PYVISIM_NUM_THREADS`` instead (can be
        overridden by setting ``os.environ["PYVISIM_NUM_THREADS"]``).
    :raises ValueError: If any parameter is outside its valid range.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        batch_size: int = 16,
        num_workers: int | None = None,
    ):
        super().__init__(batch_size=batch_size)
        _validate_window(window_size, sigma)
        _validate_stabilizers(k1, k2)
        self._window_size = window_size
        self._sigma = sigma
        self._k1 = k1
        self._k2 = k2
        self._kernel = gaussian_kernel(window_size, sigma)
        # float32 copy for the compiled kernel, cast once instead of per call.
        self._kernel32 = self._kernel.astype(np.float32)
        self._num_workers = (
            num_workers if num_workers is not None else get_kernel_threads()
        )

    @property
    def window_size(self) -> int:
        """Side length of the Gaussian window, in pixels."""
        return self._window_size

    @property
    def sigma(self) -> float:
        """Standard deviation of the Gaussian window."""
        return self._sigma

    @property
    def k1(self) -> float:
        """Luminance stabilization constant."""
        return self._k1

    @property
    def k2(self) -> float:
        """Contrast stabilization constant."""
        return self._k2

    def _validate_image_shape(self, height: int, width: int) -> None:
        """
        Reject images smaller than the Gaussian window.

        :param height: Height of the images, in pixels.
        :param width: Width of the images, in pixels.
        :raises ValueError: If the window does not fit inside the images.
        """
        if min(height, width) < self._window_size:
            raise ValueError(
                f"Images of size {height}x{width} are smaller than the "
                f"{self._window_size}x{self._window_size} SSIM window. Use "
                "larger images or a smaller window_size."
            )

    def _score_pairs(
        self, images1: Float64NumpyArray, images2: Float64NumpyArray
    ) -> Float64NumpyArray:
        """
        Compute the mean SSIM of each aligned image pair.

        The pairs are scored by the compiled fused window kernel in
        ``float32`` precision, one channel plane at a time, and the SSIM map
        mean is pooled over all channels.

        :param images1: ``(B, H, W, C)`` ``float64`` batch, one image per pair.
        :param images2: ``(B, H, W, C)`` ``float64`` batch, aligned with
            ``images1``.
        :return: A ``(B,)`` array of mean SSIM scores.
        """
        c1 = (self._k1 * CANONICAL_DATA_RANGE) ** 2
        c2 = (self._k2 * CANONICAL_DATA_RANGE) ** 2
        n_pairs, height, width, n_channels = images1.shape
        sum_l_cs, _ = ssim_plane_sums(
            _as_planes(images1),
            _as_planes(images2),
            self._kernel32,
            c1,
            c2,
            self._num_workers,
        )
        map_size = (height - self._window_size + 1) * (width - self._window_size + 1)
        scores: Float64NumpyArray = sum_l_cs.reshape(n_pairs, n_channels).sum(
            axis=1
        ) / (n_channels * map_size)
        return scores

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(window_size={self._window_size}, "
            f"sigma={self._sigma}, k1={self._k1}, k2={self._k2}, "
            f"batch_size={self.batch_size})"
        )
