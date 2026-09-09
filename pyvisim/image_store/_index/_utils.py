"""
Shared array plumbing for the search indexes.

The helpers here validate and shape the matrices that go into an index and come
back out of it: the gallery matrix an index takes ownership of, the query batch
handed to a nearest-neighbour search, and the fixed-width result arrays a search
returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray

#: Metric spaces the compiled indexes can be built for.
SUPPORTED_SPACES = ("cosine", "l2", "ip")

#: Literal alias for the accepted ``space`` argument.
Space = Literal["cosine", "l2", "ip"]


def validate_space(space: str) -> None:
    """
    Reject a metric space the compiled indexes cannot be built for.

    :param space: The requested metric space.
    :raises ValueError: If ``space`` is not one of :data:`SUPPORTED_SPACES`.
    """
    if space not in SUPPORTED_SPACES:
        raise ValueError(
            f"Unsupported space {space!r}. Supported spaces are: "
            f"{sorted(SUPPORTED_SPACES)}."
        )


def as_gallery_matrix(vectors: FloatNumpyArray) -> Float32NumpyArray:
    """
    Copy the gallery vectors into a contiguous float32 matrix.

    The copy is what the index takes ownership of, so a later mutation of the
    caller's array cannot desynchronise the index from its vectors.

    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :return: A fresh contiguous ``(N, D)`` float32 matrix.
    :raises ValueError: If the vectors are not a non-empty 2-D matrix.
    """
    # An unconditional copy: 'ascontiguousarray' would hand back the caller's
    # own array whenever it already is a C-contiguous float32 matrix, and the
    # index would then lock and normalise it in place.
    gallery = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if gallery.ndim != 2:
        raise ValueError(
            "Gallery embeddings must form a 2-D (num_vectors, dim) matrix, got "
            f"{gallery.ndim} dimension(s)."
        )
    if gallery.shape[0] == 0:
        raise ValueError("Cannot build an index from an empty gallery.")
    return gallery


def as_read_only(matrix: Float32NumpyArray) -> Float32NumpyArray:
    """
    Mark a matrix as read-only so its owner stays the only writer.

    :param matrix: The matrix to lock.
    :return: The same matrix, no longer writeable.
    """
    matrix.flags.writeable = False
    return matrix


def as_decoded_gallery(vectors: FloatNumpyArray) -> Float32NumpyArray:
    """
    Shape the vectors an index decoded into the matrix it hands out.

    :param vectors: The ``(N, D)`` block an index read back out of its own
        storage.
    :return: The block as a contiguous read-only float32 matrix.
    """
    return as_read_only(np.ascontiguousarray(vectors, dtype=np.float32))


def as_query_matrix(query_vectors: FloatNumpyArray, dim: int) -> Float32NumpyArray:
    """
    Shape a query argument into a contiguous ``(M, D)`` float32 batch.

    :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of queries.
    :param dim: Dimensionality the index was built for.
    :return: The queries as a contiguous ``(M, D)`` float32 matrix.
    :raises ValueError: If the queries are neither 1-D nor 2-D, or their
        dimensionality does not match the index.
    """
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2:
        raise ValueError(
            "Query vectors must be a (D,) vector or an (M, D) matrix, got "
            f"{queries.ndim} dimension(s)."
        )
    if queries.shape[1] != dim:
        raise ValueError(
            f"Query vectors have dimensionality {queries.shape[1]}, but the "
            f"index was built for {dim}."
        )
    return queries


def as_id_array(ids: Sequence[int] | IntNumpyArray, num_vectors: int) -> IntNumpyArray:
    """
    Shape the row numbers handed to a vector lookup into a validated array.

    :param ids: Gallery row numbers, shape ``(n,)``, at least one.
    :param num_vectors: Number of vectors the index holds.
    :return: The row numbers as a contiguous ``intp`` array.
    :raises ValueError: If ``ids`` is empty, not one-dimensional, holds
        non-integers, or names a row outside ``[0, num_vectors)``.
    """
    rows = np.asarray(ids)
    if rows.ndim != 1:
        raise ValueError(
            f"'ids' must be a 1-D sequence of row numbers, got {rows.ndim} "
            f"dimension(s)."
        )
    if rows.size == 0:
        raise ValueError("'ids' must name at least one row, got none.")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(
            f"'ids' must hold integer row numbers, got dtype {rows.dtype}."
        )
    lowest, highest = int(rows.min()), int(rows.max())
    if lowest < 0 or highest >= num_vectors:
        raise ValueError(
            f"'ids' must lie in [0, {num_vectors}), got values from {lowest} to "
            f"{highest}."
        )
    return np.ascontiguousarray(rows, dtype=np.intp)


def is_explicit_thread_count(num_threads: int) -> bool:
    """
    Report whether a thread count overrides the index's own default.

    The compiled indexes already default to one thread per available core, and
    they read a non-positive count as "unset" only in some of their entry
    points, so anything but a positive count is left to them.

    :param num_threads: The requested thread count.
    :return: ``True`` if the count should be pushed into the index.
    """
    return num_threads > 0


def validate_k(k: int) -> int:
    """
    Reject a non-positive neighbour count.

    :param k: Number of nearest neighbours requested per query.
    :return: ``k`` as an integer.
    :raises ValueError: If ``k`` is not a positive integer.
    """
    if k < 1:
        raise ValueError(f"'k' must be a positive integer, got {k}.")
    return int(k)


def pad_results(
    labels: IntNumpyArray,
    distances: FloatNumpyArray,
    k: int,
) -> tuple[Float32NumpyArray, IntNumpyArray]:
    """
    Widen a result block to ``k`` columns, marking the missing neighbours.

    A gallery smaller than ``k`` yields fewer neighbours than were asked for.
    The free columns are filled with the ``-1`` id and an infinite distance, so
    callers always receive the ``(M, k)`` block they requested.

    :param labels: The ``(M, n)`` gallery ids returned by the index.
    :param distances: The ``(M, n)`` distances returned by the index.
    :param k: Number of columns the result must have.
    :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays.
    """
    scores = np.asarray(distances, dtype=np.float32)
    ids = np.asarray(labels, dtype=np.intp)
    missing = k - scores.shape[1]
    if missing > 0:
        rows = scores.shape[0]
        scores = np.hstack([scores, np.full((rows, missing), np.inf, np.float32)])
        ids = np.hstack([ids, np.full((rows, missing), -1, np.intp)])
    return scores, ids
