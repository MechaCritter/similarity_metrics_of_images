"""Adapter for search indexes built outside of this library."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ._utils import (
    as_gallery_matrix,
    as_id_array,
    as_query_matrix,
    as_read_only,
    validate_k,
)

#: Name reported by an index the caller did not name.
DEFAULT_EXTERNAL_NAME = "external"


class ExternalSearchIndex:
    """
    A search index built elsewhere, adapted to the store's interface.

    Whether a score is a distance (lower is better) or a
    similarity (higher is better) depends on the metric the wrapped
    index was built for, and normalising the vectors for a cosine ranking is the
    caller's job, before the index is built.

    :param index: The index to search through. It must expose a
        ``search(queries, k)`` returning a ``(scores, ids)`` pair of ``(M, k)``
        arrays whose ids are row numbers into ``vectors``.
    :param vectors: The gallery vectors the index was built over, shape
        ``(N, D)``, in the order its ids refer to.
    :param name: Name identifying the index, kept across a save/load round trip
        so a store can be rebuilt on a matching one. Defaults to
        :data:`DEFAULT_EXTERNAL_NAME`.
    :raises AttributeError: If ``index`` has no ``search`` method.
    :raises ValueError: If ``vectors`` is not a non-empty 2-D matrix, or the
        index reports a size that does not match it.
    """

    def __init__(
        self,
        index: Any,
        vectors: FloatNumpyArray,
        *,
        name: str | None = None,
    ) -> None:
        if not callable(getattr(index, "search", None)):
            raise AttributeError(
                f"{type(index).__name__} has no 'search' method, so it cannot be "
                f"used as a search index."
            )

        self._index = index
        self._name = DEFAULT_EXTERNAL_NAME if name is None else str(name)
        self._vectors = as_read_only(as_gallery_matrix(vectors))

        indexed = getattr(index, "ntotal", None)
        if indexed is not None and int(indexed) != self._vectors.shape[0]:
            raise ValueError(
                f"The index holds {int(indexed)} vectors, but {self._vectors.shape[0]} "
                f"were passed alongside it."
            )

    @classmethod
    def from_faiss_index(
        cls,
        index: Any,
        vectors: FloatNumpyArray | None = None,
        *,
        name: str | None = None,
    ) -> ExternalSearchIndex:
        """
        Adapt a FAISS index, reading its vectors back when it can produce them.

        An index that cannot reconstruct needs its vectors passed explicitly; so
        does one whose reconstruction this cannot catch, since a few index
        types abort the process instead of raising an error.

        Normalization, if any, must be done by the caller. An index built for
        ``METRIC_INNER_PRODUCT`` only ranks by cosine similarity if the vectors
        were L2-normalised before they were added, and the queries handed to
        :meth:`search` must be normalised the same way.

        :param index: The FAISS index to search through.
        :param vectors: The gallery vectors the index was built over, shape
            ``(N, D)``. Reconstructed from the index when omitted.
        :param name: Name identifying the index, defaults to
            :data:`DEFAULT_EXTERNAL_NAME`.
        :return: An :class:`ExternalSearchIndex` around ``index``.
        :raises ValueError: If ``vectors`` is omitted and the index cannot
            reconstruct them, or the index reports a different size.
        """
        if vectors is None:
            vectors = _reconstruct_faiss_vectors(index)
        if vectors is None:
            raise ValueError(
                f"{type(index).__name__} cannot reconstruct its vectors, so they "
                f"must be passed explicitly: "
                f"ExternalSearchIndex.from_faiss_index(index, vectors)."
            )
        return cls(index, vectors, name=name)

    @property
    def index(self) -> Any:
        """The wrapped index, as it was passed in."""
        return self._index

    @property
    def name(self) -> str:
        """Name identifying the index across a save/load round trip."""
        return self._name

    @property
    def vectors(self) -> Float32NumpyArray:
        """The ``(N, D)`` gallery matrix the index was built over, read-only."""
        return self._vectors

    def vectors_at(self, ids: Sequence[int] | IntNumpyArray) -> Float32NumpyArray:
        """
        Read the gallery vectors stored under the given row numbers.

        :param ids: Gallery row numbers, shape ``(n,)``, at least one.
        :return: The ``(n, D)`` block of the requested vectors, read-only and in
            the given order.
        :raises ValueError: If ``ids`` is empty, not one-dimensional, holds
            non-integers, or names a row outside the gallery.
        """
        rows = as_id_array(ids, len(self))
        return as_read_only(np.ascontiguousarray(self._vectors[rows]))

    @property
    def dim(self) -> int:
        """Dimensionality of the indexed vectors."""
        return int(self._vectors.shape[1])

    def __len__(self) -> int:
        return int(self._vectors.shape[0])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"index={type(self._index).__name__}, num_vectors={len(self)}, "
            f"dim={self.dim})"
        )

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        The scores are whatever the wrapped index returns, unchanged.

        :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays. ``ids`` are
            gallery row numbers, and missing neighbours are reported as ``-1``.
        :raises ValueError: If ``k`` is not positive or the queries do not match
            the indexed dimensionality.
        """
        k = validate_k(k)
        queries = as_query_matrix(query_vectors, self.dim)
        scores, ids = self._index.search(queries, k)
        return (
            cast(Float32NumpyArray, np.asarray(scores, dtype=np.float32)),
            cast(IntNumpyArray, np.asarray(ids, dtype=np.intp)),
        )


def _reconstruct_faiss_vectors(index: Any) -> Float32NumpyArray | None:
    """Read a FAISS index's vectors back, if it is able to produce them."""
    ntotal = getattr(index, "ntotal", None)
    if ntotal is None:
        return None
    try:
        vectors = index.reconstruct_n(0, int(ntotal))
    except (AttributeError, RuntimeError, TypeError):
        # Not every index keeps its vectors around: a purely compressed index
        # has none to give back, and an IVF index needs a direct map first.
        return None
    return cast(Float32NumpyArray, np.asarray(vectors, dtype=np.float32))
