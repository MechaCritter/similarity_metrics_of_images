"""
Re-ranking of retrieval results with k-reciprocal encoding.

:class:`KReciprocalReranker` re-orders the candidates a store retrieved for a
query by how much the candidates' own neighbourhoods agree with the query's,
following Zhong et al. (CVPR 2017).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np

from ..distance import euclidean_distances
from ..typing import (
    BoolNumpyArray,
    Float64NumpyArray,
    FloatNumpyArray,
    IntNumpyArray,
)
from ._index import ExternalSearchIndex
from .candidate import Candidate
from .image_store import InMemoryImageEmbeddingStore

#: Share of a member's own reciprocal neighbourhood that must already lie in the
#: probe's for the two neighbourhoods to be merged (Eq. 4 of Zhong et al.).
_EXPANSION_OVERLAP = 2.0 / 3.0


class KReciprocalReranker:
    """
    Re-rank retrieval candidates with k-reciprocal encoding.

    Implements the re-ranking of Zhong et al. [1]. Two images are k-reciprocal
    neighbours when each ranks among the ``k1`` nearest neighbours of the
    other, a far stricter relation than plain proximity to the query: a false
    match may lie close to the query, but the query rarely lies close to the
    false match's own neighbours. The query and every candidate are encoded
    into a k-reciprocal feature, a vector over the candidate set that holds a
    Gaussian weight for each k-reciprocal neighbour and zero elsewhere, and the
    Jaccard distance between the query's feature and a candidate's says how
    much their neighbourhoods agree. The final distance the candidates are
    re-ranked by mixes that Jaccard distance with the original one.

    :param store: The store the candidates were retrieved from.
    :param k1: Size of the neighbourhoods the k-reciprocal sets are built
        from. Defaults to ``20`` as in [1].
    :param k2: Size of the neighbourhood the local query expansion averages
        the k-reciprocal features over. ``1`` turns the expansion off. Defaults
        to ``6`` as in [1].
    :param lambda_value: Weight of the original distance in the final
        distance, from ``0`` (Jaccard distance only) to ``1`` (original ranking
        kept). Defaults to ``0.3`` as in [1].
    :raises TypeError: If ``store`` is not an
        :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`.
    :raises ValueError: If the store searches through an external index,
        ``k1`` or ``k2`` is not a positive integer, ``k2`` exceeds ``k1``, or
        ``lambda_value`` lies outside ``[0, 1]``.

    Example
    -------

    >>> from pyvisim.image_store import InMemoryImageEmbeddingStore, KReciprocalReranker
    >>> from pyvisim.neural_networks import ClipEmbedder
    >>>
    >>> store = InMemoryImageEmbeddingStore(gallery_paths, ClipEmbedder(), "hnsw")
    >>> reranker = KReciprocalReranker(store, k1=20, k2=6, lambda_value=0.3)
    >>>
    >>> # Retrieve a pool of candidates, then keep the best five after re-ranking
    >>> # Retrieve more candidates than you finally want with the `InMemoryImageEmbeddingStore`,
    >>> # at least `k1` and better a few dozen more than `top_k`.
    >>> candidates = store.retrieve_top_k_similar(query_image, k=100)[0]
    >>> best = reranker.rerank(candidates, top_k=5)

    References:
    ===========
    [1] Z. Zhong, L. Zheng, D. Cao, and S. Li, "Re-ranking Person
        Re-identification with k-reciprocal Encoding," in Proc. CVPR,
        pp. 1318-1327, 2017.
    """

    def __init__(
        self,
        store: InMemoryImageEmbeddingStore,
        *,
        k1: int = 20,
        k2: int = 6,
        lambda_value: float = 0.3,
    ) -> None:
        if not isinstance(store, InMemoryImageEmbeddingStore):
            raise TypeError(
                f"'store' must be an InMemoryImageEmbeddingStore, got "
                f"{type(store).__name__}."
            )
        if isinstance(store.index, ExternalSearchIndex):
            raise ValueError(
                "The store searches through an external index, whose scores may "
                "be similarities or distances of an unknown metric. K-reciprocal "
                "re-ranking needs the distances of the store's space, which only "
                "the built-in indexes report."
            )
        if not isinstance(k1, int) or k1 < 1:
            raise ValueError(f"'k1' must be a positive integer, got {k1!r}.")
        if not isinstance(k2, int) or k2 < 1:
            raise ValueError(f"'k2' must be a positive integer, got {k2!r}.")
        if k2 > k1:
            raise ValueError(f"'k2' must not exceed 'k1', got k2={k2} and k1={k1}.")
        if (
            isinstance(lambda_value, bool)
            or not isinstance(lambda_value, numbers.Real)
            or not 0.0 <= float(lambda_value) <= 1.0
        ):
            raise ValueError(
                f"'lambda_value' must be a number in [0, 1], got {lambda_value!r}."
            )
        self._store: InMemoryImageEmbeddingStore = store
        self._k1: int = int(k1)
        self._k2: int = int(k2)
        self._lambda_value: float = float(lambda_value)

    @property
    def store(self) -> InMemoryImageEmbeddingStore:
        """The store the candidates are read back from."""
        return self._store

    @property
    def k1(self) -> int:
        """Size of the neighbourhoods the k-reciprocal sets are built from."""
        return self._k1

    @property
    def k2(self) -> int:
        """Size of the neighbourhood of the local query expansion."""
        return self._k2

    @property
    def lambda_value(self) -> float:
        """Weight of the original distance in the final distance."""
        return self._lambda_value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(k1={self._k1}, k2={self._k2}, "
            f"lambda_value={self._lambda_value})"
        )

    def rerank(self, candidates: Sequence[Candidate], top_k: int) -> list[Candidate]:
        """
        Re-rank the candidates of one query and return the best ``top_k``.

        The candidates are the ranked matches of a single query, as one row of
        :meth:`~pyvisim.image_store.InMemoryImageEmbeddingStore.retrieve_top_k_similar`
        returns them, and their scores are the query's distances to them. The
        neighbourhoods are built among the candidates themselves, so a pool no
        larger than the answer leaves them nothing to say: retrieve at least
        ``k1`` candidates and a few dozen more than ``top_k``. Neighbourhood
        sizes beyond the number of candidates are capped at it.

        :param candidates: The ranked matches of one query, at least one, no
            path twice.
        :param top_k: Number of best re-ranked candidates to return. More than
            there are candidates returns them all.
        :return: The ``top_k`` best candidates, best first, each scored by the
            final distance of Eq. (12) of [1], which lies in ``[0, 1]`` and is
            lower for a better match.
        :raises TypeError: If an element of ``candidates`` is not a
            :class:`~pyvisim.image_store.Candidate`.
        :raises ValueError: If ``candidates`` is empty, names a path twice or
            one the store does not hold, a score is not finite, or ``top_k``
            is not a positive integer.
        """
        _validate_candidates(candidates)
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
            raise ValueError(f"'top_k' must be a positive integer, got {top_k!r}.")
        if top_k < 1:
            raise ValueError(f"'top_k' must be a positive integer, got {top_k!r}.")
        pool = len(candidates)
        k1 = min(self._k1, pool)
        final = _k_reciprocal_distances(
            self._distance_matrix(candidates), k1, min(self._k2, k1), self._lambda_value
        )
        order = np.argsort(final, kind="stable")[:top_k]
        return [Candidate(candidates[int(i)].path, float(final[i])) for i in order]

    def _distance_matrix(self, candidates: Sequence[Candidate]) -> Float64NumpyArray:
        """
        Lay out the pairwise distances of the probe and the candidates.

        Row and column ``0`` are the probe. Its distances to the candidates are
        their scores, and the distances among the candidates are computed in
        the store's space from their embeddings, so the whole matrix speaks the
        metric the scores were ranked by.

        :param candidates: The ranked matches of one query.
        :return: The symmetric ``(n + 1, n + 1)`` distance matrix.
        :raises ValueError: If a candidate's path is not in the store.
        """
        vectors = self._store.embeddings_of(
            [candidate.path for candidate in candidates]
        )
        scores = np.array(
            [candidate.score for candidate in candidates], dtype=np.float64
        )
        size = len(candidates) + 1
        matrix = np.zeros((size, size), dtype=np.float64)
        matrix[0, 1:] = scores
        matrix[1:, 0] = scores
        matrix[1:, 1:] = _pairwise_distances(vectors, self._store.space)
        return matrix


def _validate_candidates(candidates: Sequence[Candidate]) -> None:
    """
    Reject a candidate list the re-ranking cannot run on.

    :param candidates: The ranked matches of one query.
    :raises TypeError: If an element is not a :class:`Candidate`.
    :raises ValueError: If the list is empty, a score is not finite, or a path
        appears twice.
    """
    if len(candidates) == 0:
        raise ValueError("'candidates' must hold at least one candidate, got none.")
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise TypeError(
                f"'candidates' must hold Candidate objects, got "
                f"{type(candidate).__name__}."
            )
        if not math.isfinite(candidate.score):
            raise ValueError(
                f"The score of {candidate.path!r} must be finite, got "
                f"{candidate.score!r}."
            )
    paths = [candidate.path for candidate in candidates]
    if len(set(paths)) != len(paths):
        raise ValueError("'candidates' must not name a path twice.")


def _pairwise_distances(vectors: FloatNumpyArray, space: str) -> Float64NumpyArray:
    """
    Compute the distances among the candidates the way the store's index does.

    :param vectors: The ``(n, D)`` candidate embeddings, as the index stores
        them.
    :param space: The store's metric space.
    :return: The symmetric ``(n, n)`` distance matrix with an exactly zero
        diagonal.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if space == "l2":
        distances = euclidean_distances(matrix, matrix) ** 2
    else:
        # The cosine space stores unit vectors, so "1 - inner product" is the
        # cosine distance there and the inner-product distance in "ip" space.
        distances = 1.0 - matrix @ matrix.T
    np.fill_diagonal(distances, 0.0)
    return distances


def _reciprocal_neighbours(ranking: IntNumpyArray, k: int) -> BoolNumpyArray:
    """
    Mark the k-reciprocal neighbours of every element, Eq. (3) of Zhong et al.

    The neighbourhood ``N(p, k)`` of Eq. (2) is read off the ranking list of
    the whole set the way the authors' implementation does, the element itself
    included at rank zero, hence ``k + 1`` entries.

    :param ranking: Every element's ranking list of the set, itself first.
    :param k: Neighbourhood size, without the element itself.
    :return: A boolean matrix whose ``[p, g]`` is ``True`` when ``p`` and ``g``
        rank among each other's ``k`` nearest neighbours.
    """
    size = ranking.shape[0]
    neighbours = np.zeros((size, size), dtype=bool)
    neighbours[np.arange(size)[:, np.newaxis], ranking[:, : k + 1]] = True
    reciprocal: BoolNumpyArray = neighbours & neighbours.T
    return reciprocal


def _expanded_reciprocal_neighbours(ranking: IntNumpyArray, k: int) -> BoolNumpyArray:
    """
    Expand the k-reciprocal neighbourhoods, Eq. (4) of Zhong et al.

    A member ``q`` of ``R(p, k)`` brings its own ``R(q, k/2)`` into the
    neighbourhood of ``p`` when at least two thirds of that set already lie in
    ``R(p, k)``.

    :param ranking: Every element's ranking list of the set, itself first.
    :param k: Size of the neighbourhoods the reciprocal sets are built from.
    :return: A boolean matrix whose ``[p, g]`` is ``True`` when ``g`` belongs to
        the expanded neighbourhood ``R*(p, k)``.
    """
    reciprocal = _reciprocal_neighbours(ranking, k)
    half = _reciprocal_neighbours(ranking, int(np.around(k / 2)))
    # overlap[p, q] counts the members of R(q, k/2) that R(p, k) already holds.
    overlap = reciprocal.astype(np.int64) @ half.T.astype(np.int64)
    accepted = reciprocal & (
        overlap >= _EXPANSION_OVERLAP * half.sum(axis=1)[np.newaxis, :]
    )
    merged = (accepted.astype(np.int64) @ half.astype(np.int64)) > 0
    expanded: BoolNumpyArray = reciprocal | merged
    return expanded


def _k_reciprocal_distances(
    distances: FloatNumpyArray,
    k1: int,
    k2: int,
    lambda_value: float,
) -> Float64NumpyArray:
    """
    Compute the final re-ranking distances from the probe to every candidate.

    Follows Section 3 of Zhong et al. over the set made of the probe and the
    candidates, the probe being row and column ``0``.

    :param distances: The symmetric ``(n + 1, n + 1)`` matrix of pairwise
        distances, the probe first.
    :param k1: Size of the neighbourhoods the k-reciprocal sets are built
        from, at most ``n``.
    :param k2: Size of the neighbourhood of the local query expansion, at most
        ``k1``.
    :param lambda_value: Weight of the original distance in the final one.
    :return: The ``(n,)`` final distances from the probe to the candidates, in
        candidate order.
    """
    # An index reports the distance of an exact duplicate as a tiny negative
    # number of rounding, and a distance below zero means nothing here, so the
    # matrix is floored at zero.
    floored = np.maximum(np.asarray(distances, dtype=np.float64), 0.0)
    # The paper leaves the scale of the original distance open, and the
    # authors' implementation divides every row by its largest entry, so that
    # the Gaussian kernel of Eq. (7) and the mix of Eq. (12) see distances in
    # [0, 1] whatever the metric.
    scale = floored.max(axis=1, keepdims=True)
    scale[scale <= 0.0] = 1.0
    scaled = floored / scale
    # Every element's ranking list of the whole set, itself first at distance 0.
    ranking = np.argsort(scaled, axis=1, kind="stable")
    # Eq. (7): the k-reciprocal feature is a Gaussian kernel of the original
    # distance over the expanded neighbourhood R*(p, k1) and zero elsewhere.
    # Each row is scaled to unit L1 norm as in the authors' implementation, so
    # that the Jaccard distance below compares the shape of two neighbourhoods
    # rather than their size.
    features = np.where(
        _expanded_reciprocal_neighbours(ranking, k1), np.exp(-scaled), 0.0
    )
    features /= features.sum(axis=1, keepdims=True)
    # Eq. (11): local query expansion, the mean feature over the k2 nearest
    # neighbours, the element itself included as in the authors' implementation.
    if k2 > 1:
        features = features[ranking[:, :k2]].mean(axis=1)
    # Eq. (10): the Jaccard distance between the probe's feature and every
    # other one, written with the element-wise minimum and maximum.
    probe = features[0]
    jaccard = 1.0 - np.minimum(probe, features).sum(axis=1) / np.maximum(
        probe, features
    ).sum(axis=1)
    # Eq. (12): the final distance mixes the Jaccard and the original distance.
    final = (1.0 - lambda_value) * jaccard + lambda_value * scaled[0]
    return np.asarray(final[1:], dtype=np.float64)
