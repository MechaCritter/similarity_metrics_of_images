"""
Pairwise distance and similarity metrics implemented in pure NumPy.

This module hosts pyvisim's own implementations of the vector metrics used to
compare image embeddings: :func:`cosine_similarity`, :func:`euclidean_distances`
and :func:`manhattan_distances`. All three operate on 2-D ``(N, D)`` matrices
and return a full ``(N, M)`` pairwise result computed with vectorized NumPy
operations only.

The higher-level, name-based metric registry lives in ``pyvisim._utils`` and
is exposed through the ``similarity_func`` argument of the embedders.
"""

from __future__ import annotations

import numpy as np

from .typing import Float64NumpyArray, FloatNumpyArray

__all__ = ["cosine_similarity", "euclidean_distances", "manhattan_distances"]

#: Default upper bound on the size of the broadcast temporary used by
#: :func:`manhattan_distances`, mirroring scikit-learn's chunked
#: working-memory strategy for pairwise distances. Manhattan is the only
#: metric that materializes an ``(N, M, D)`` intermediate; cosine and
#: Euclidean reduce to matrix products whose temporaries never exceed the
#: inputs and the ``(N, M)`` result itself.
_WORKING_MEMORY_BYTES = 256 * 1024**2


def _validate_pairwise_inputs(
    x: FloatNumpyArray, y: FloatNumpyArray
) -> tuple[Float64NumpyArray, Float64NumpyArray]:
    """
    Coerce a pair of metric inputs into compatible 2-D float64 matrices.

    Computing in float64 sidesteps the catastrophic cancellation that the
    dot-product expansion of the Euclidean distance suffers from in float32.

    :param x: First matrix of shape ``(N, D)``.
    :param y: Second matrix of shape ``(M, D)``.
    :return: The validated matrices as float64 arrays.
    :raises ValueError: If either input is not 2-D or the feature dimensions
        do not match.
    """
    x64: Float64NumpyArray = np.asarray(x, dtype=np.float64)
    y64: Float64NumpyArray = np.asarray(y, dtype=np.float64)
    if x64.ndim != 2 or y64.ndim != 2:
        raise ValueError(
            f"Pairwise metrics expect 2-D arrays of shape (N, D). "
            f"Got {x64.ndim}-D and {y64.ndim}-D inputs."
        )
    if x64.shape[1] != y64.shape[1]:
        raise ValueError(
            f"Incompatible feature dimensions: x has {x64.shape[1]} features "
            f"but y has {y64.shape[1]}."
        )
    return x64, y64


def _squared_row_norms(x: Float64NumpyArray) -> Float64NumpyArray:
    """
    Compute ``||x_i||^2`` for every row of ``x``.

    ``einsum`` contracts each row with itself directly, avoiding the ``(N, D)``
    temporary that ``(x ** 2).sum(axis=1)`` would allocate.

    :param x: Matrix of shape ``(N, D)``.
    :return: Vector of squared row norms of shape ``(N,)``.
    """
    squared_norms: Float64NumpyArray = np.einsum("ij,ij->i", x, x)
    return squared_norms


def _safe_row_norms(x: Float64NumpyArray) -> Float64NumpyArray:
    """
    Compute the L2 norm of every row of ``x``.

    Zero norms are substituted with 1 so that all-zero rows divide cleanly
    and yield a similarity of 0 instead of NaN.

    :param x: Matrix of shape ``(N, D)``.
    :return: Vector of row norms of shape ``(N,)``.
    """
    norms = np.sqrt(_squared_row_norms(x))
    norms[norms == 0.0] = 1.0
    return norms


def cosine_similarity(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the pairwise cosine similarity between two matrices.

    The raw inner products are computed with a single matrix product and the
    row norms are divided out of the ``(N, M)`` result in place.

    :param x: First matrix of shape ``(N, D)``.
    :param y: Second matrix of shape ``(M, D)``.
    :return: Cosine similarity matrix of shape ``(N, M)``.
    :raises ValueError: If either input is not 2-D or the feature dimensions
        do not match.
    """
    x, y = _validate_pairwise_inputs(x, y)
    x_norms = _safe_row_norms(x)
    y_norms = x_norms if y is x else _safe_row_norms(y)
    similarities: Float64NumpyArray = x @ y.T
    similarities /= x_norms[:, np.newaxis]
    similarities /= y_norms[np.newaxis, :]
    return similarities


def euclidean_distances(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the pairwise Euclidean (L2) distance between two matrices.

    Uses the expansion ``||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2`` so the
    whole distance matrix reduces to one matrix product plus two rank-1
    updates. Lower values mean more similar.

    :param x: First matrix of shape ``(N, D)``.
    :param y: Second matrix of shape ``(M, D)``.
    :return: Euclidean distance matrix of shape ``(N, M)``.
    :raises ValueError: If either input is not 2-D or the feature dimensions
        do not match.
    """
    x, y = _validate_pairwise_inputs(x, y)
    distances = x @ y.T
    distances *= -2.0
    distances += _squared_row_norms(x)[:, np.newaxis]
    distances += _squared_row_norms(y)[np.newaxis, :]
    # Rounding in the expansion can push exact zeros slightly negative.
    np.maximum(distances, 0.0, out=distances)
    if y is x:
        # Self-distances are exactly zero by definition.
        np.fill_diagonal(distances, 0.0)
    result: Float64NumpyArray = np.sqrt(distances, out=distances)
    return result


def manhattan_distances(
    x: FloatNumpyArray,
    y: FloatNumpyArray,
    *,
    working_memory_bytes: int | None = None,
) -> Float64NumpyArray:
    """
    Compute the pairwise Manhattan (L1) distance between two matrices.

    The broadcast difference materializes a ``(chunk, M, D)`` temporary, so
    rows of ``x`` are processed in chunks that keep that temporary under the
    working-memory budget. Lower values mean more similar.

    :param x: First matrix of shape ``(N, D)``.
    :param y: Second matrix of shape ``(M, D)``.
    :param working_memory_bytes: Cap on the size of the broadcast temporary.
        Defaults to 256 MiB. The chunking only changes memory usage, never
        the result.
    :return: Manhattan distance matrix of shape ``(N, M)``.
    :raises ValueError: If either input is not 2-D, the feature dimensions
        do not match, or ``working_memory_bytes`` is not positive.
    """
    if working_memory_bytes is None:
        working_memory_bytes = _WORKING_MEMORY_BYTES
    elif working_memory_bytes <= 0:
        raise ValueError(
            f"working_memory_bytes must be positive. Got {working_memory_bytes}."
        )
    x, y = _validate_pairwise_inputs(x, y)
    n_rows = x.shape[0]
    bytes_per_row = y.size * np.float64().itemsize
    chunk_size = max(1, working_memory_bytes // max(1, bytes_per_row))
    distances = np.empty((n_rows, y.shape[0]), dtype=np.float64)
    for start in range(0, n_rows, chunk_size):
        stop = start + chunk_size
        np.sum(
            np.abs(x[start:stop, np.newaxis, :] - y[np.newaxis, :, :]),
            axis=-1,
            out=distances[start:stop],
        )
    return distances
