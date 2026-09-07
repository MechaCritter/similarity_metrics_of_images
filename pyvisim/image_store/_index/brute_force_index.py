"""Exhaustive nearest-neighbour search over a gallery."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ._bindings import _hnswlib
from ._params import BRUTE_FORCE_TO_HNSWLIB, as_backend_params
from ._utils import (
    Space,
    as_decoded_gallery,
    as_gallery_matrix,
    as_query_matrix,
    is_explicit_thread_count,
    pad_results,
    validate_k,
    validate_space,
)


class BruteForceIndex(_hnswlib.BFIndex):
    """
    Exhaustive (brute-force) index scanning the whole gallery per query.

    Every query is compared against every gallery vector, so the ranking is
    exact at a cost linear in the gallery size. It is the right choice for
    galleries small enough to scan and the baseline to measure an approximate
    index against.

    The index owns the gallery vectors: they are copied into it at construction
    time and read back on demand, so no second copy is held.

    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param space: Metric space to search in, ``"cosine"``, ``"l2"`` or ``"ip"``.
        In cosine space the vectors are stored L2-normalised.
    :param num_threads: Threads used to run batched queries. ``-1`` uses every
        available core.
    :raises ValueError: If ``space`` is unknown or the gallery is not a
        non-empty 2-D matrix.
    """

    def __init__(
        self,
        vectors: FloatNumpyArray,
        *,
        space: Space = "cosine",
        num_threads: int = -1,
    ) -> None:
        validate_space(space)
        gallery = as_gallery_matrix(vectors)
        super().__init__(space, gallery.shape[1])

        self._num_vectors = int(gallery.shape[0])
        self.init_index(**self._backend_build_params())
        if is_explicit_thread_count(num_threads):
            self.set_num_threads(int(num_threads))
        self.add_items(gallery, np.arange(self._num_vectors))

    def _backend_build_params(self) -> dict[str, Any]:
        """The build parameters of the current gallery, in backend keywords."""
        return as_backend_params(
            {"capacity": self._num_vectors}, BRUTE_FORCE_TO_HNSWLIB
        )

    @property
    def vectors(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery matrix, read back from the index.

        The index stores each vector next to its id, so every access copies them
        out into a fresh read-only array. In cosine space they come back
        L2-normalised, the form they were indexed in.
        """
        decoded = self.get_items(np.arange(self._num_vectors), return_type="numpy")
        return as_decoded_gallery(decoded)

    def __len__(self) -> int:
        return self._num_vectors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_vectors={len(self)}, dim={self.dim}, "
            f"space={self.space!r})"
        )

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        The scores are the distances of the index's metric space, so lower is
        always more similar: ``1 - cosine_similarity`` in cosine space,
        ``1 - inner_product`` in ``"ip"`` space and the squared Euclidean
        distance in ``"l2"`` space.

        :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays. ``ids`` are
            gallery row numbers; a gallery holding fewer than ``k`` vectors pads
            the free columns with the id ``-1``.
        :raises ValueError: If ``k`` is not positive or the queries do not match
            the indexed dimensionality.
        """
        k = validate_k(k)
        queries = as_query_matrix(query_vectors, self.dim)
        labels, distances = self.knn_query(queries, k=min(k, len(self)))
        return pad_results(cast(IntNumpyArray, labels), distances, k)

    def update(self, vectors: FloatNumpyArray) -> None:
        """
        Replace the gallery and rebuild the index from scratch.

        :param vectors: The new gallery matrix, shape ``(N, D)``. Its
            dimensionality must match the one the index was built for.
        :raises ValueError: If the new gallery is not a non-empty 2-D matrix or
            its dimensionality differs from the indexed one.
        """
        gallery = as_gallery_matrix(vectors)
        if gallery.shape[1] != self.dim:
            raise ValueError(
                f"New vectors have dimensionality {gallery.shape[1]}, but the "
                f"index was built for {self.dim}."
            )
        self._num_vectors = int(gallery.shape[0])
        self.reset_index()
        self.init_index(**self._backend_build_params())
        self.add_items(gallery, np.arange(self._num_vectors))
