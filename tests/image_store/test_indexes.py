"""Tests for the search indexes in :mod:`pyvisim.image_store`."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyvisim.image_store import BruteForceIndex, HnswIndex
from pyvisim.image_store._index import (
    BRUTE_FORCE_TO_HNSWLIB,
    HNSW_TO_HNSWLIB,
    accepted_index_params,
)
from pyvisim.image_store._index._bindings import _hnswlib


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    """A ``(40, 12)`` gallery matrix of well-separated random vectors.

    :returns: a float32 matrix usable by both index types.
    """
    rng = np.random.default_rng(0)
    return rng.random((40, 12), dtype=np.float32)


# Construction and exposed state


def test_indexes_extend_the_compiled_classes(vectors: np.ndarray) -> None:
    """Both indexes are the compiled hnswlib structures themselves."""
    assert isinstance(HnswIndex(vectors), _hnswlib.Index)
    assert isinstance(BruteForceIndex(vectors), _hnswlib.BFIndex)


def test_indexes_share_no_python_base() -> None:
    """The two indexes share no ancestor defined by this library.

    What they have in common is the compiled machinery every pybind11 class
    carries, not a base class of their own.
    """
    shared = set(HnswIndex.__mro__) & set(BruteForceIndex.__mro__)
    assert not [base for base in shared if base.__module__.startswith("pyvisim")]


def test_reports_size_dim_and_space(vectors: np.ndarray) -> None:
    """An index reports its gallery size, dimensionality and metric space."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        assert len(index) == vectors.shape[0]
        assert index.dim == vectors.shape[1]
        assert index.space == "cosine"


def test_repr_names_the_index(vectors: np.ndarray) -> None:
    """``repr`` names the index class and its configuration."""
    assert "HnswIndex(" in repr(HnswIndex(vectors, graph_degree=8))
    assert "graph_degree=8" in repr(HnswIndex(vectors, graph_degree=8))
    assert "BruteForceIndex(" in repr(BruteForceIndex(vectors))


def test_hnsw_exposes_its_graph_parameters(vectors: np.ndarray) -> None:
    """The HNSW build and query parameters are readable after construction."""
    index = HnswIndex(
        vectors,
        graph_degree=8,
        build_candidates=64,
        search_candidates=17,
        random_seed=3,
    )
    assert index.graph_degree == 8
    assert index.build_candidates == 64
    assert index.search_candidates == 17
    assert index.random_seed == 3


def test_hnsw_params_reach_the_backend_under_its_own_names(
    vectors: np.ndarray,
) -> None:
    """Each parameter drives the hnswlib keyword its table maps it onto."""
    index = HnswIndex(vectors, graph_degree=8, build_candidates=64)
    assert index.M == 8
    assert index.ef_construction == 64
    assert index.max_elements == vectors.shape[0]


def test_hnsw_rejects_the_backends_own_parameter_names(vectors: np.ndarray) -> None:
    """The hnswlib spellings are not accepted; only this library's names are."""
    for name in ("m", "M", "ef", "ef_search", "max_elements"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            HnswIndex(vectors, **{name: 8})


# Parameter tables


def test_tables_cover_every_constructor_parameter() -> None:
    """No parameter an index takes is missing from its table."""
    for index_cls, table in (
        (HnswIndex, HNSW_TO_HNSWLIB),
        (BruteForceIndex, BRUTE_FORCE_TO_HNSWLIB),
    ):
        taken = set(inspect.signature(index_cls.__init__).parameters)
        assert taken - {"self", "vectors"} <= set(table)


def test_accepted_params_exclude_the_ones_the_store_supplies() -> None:
    """``space`` and ``capacity`` are set by the store, not by the caller."""
    assert accepted_index_params(HNSW_TO_HNSWLIB) == (
        "graph_degree",
        "build_candidates",
        "search_candidates",
        "random_seed",
        "num_threads",
    )
    assert accepted_index_params(BRUTE_FORCE_TO_HNSWLIB) == ("num_threads",)


@pytest.mark.parametrize("space", ["cosine", "l2", "ip"])
def test_every_space_builds_and_searches(vectors: np.ndarray, space: str) -> None:
    """Each supported metric space builds a searchable index."""
    for index in (
        HnswIndex(vectors, space=space),  # type: ignore[arg-type]
        BruteForceIndex(vectors, space=space),  # type: ignore[arg-type]
    ):
        scores, ids = index.search(vectors[:2], k=3)
        assert scores.shape == (2, 3)
        assert ids.shape == (2, 3)


# Vector ownership


def test_vectors_are_read_only(vectors: np.ndarray) -> None:
    """The gallery an index hands out cannot be written to."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        with pytest.raises(ValueError, match="read-only"):
            index.vectors[0, 0] = 1.0


def test_construction_leaves_the_callers_array_alone(vectors: np.ndarray) -> None:
    """The index copies the gallery instead of adopting the caller's array."""
    original = vectors.copy()
    HnswIndex(vectors)
    BruteForceIndex(vectors)
    assert vectors.flags.writeable
    assert np.array_equal(vectors, original)


def test_cosine_index_stores_normalized_vectors(vectors: np.ndarray) -> None:
    """In cosine space the stored gallery is L2-normalised."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        norms = np.linalg.norm(index.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_indexes_decode_a_fresh_array_per_access(vectors: np.ndarray) -> None:
    """An index decodes its vectors anew on every access.

    Both indexes keep the gallery in their own storage and no second copy
    beside it, so the matrix they hand out is decoded on demand.
    """
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        assert index.vectors is not index.vectors
        assert np.array_equal(index.vectors, index.vectors)


def test_indexes_return_the_gallery_they_were_given(vectors: np.ndarray) -> None:
    """In a space that stores the vectors as given, they come back unchanged."""
    for index in (HnswIndex(vectors, space="l2"), BruteForceIndex(vectors, space="l2")):
        assert np.allclose(index.vectors, vectors, atol=1e-6)


def test_both_indexes_store_the_same_gallery(vectors: np.ndarray) -> None:
    """The two indexes agree on the vectors they were given."""
    assert np.allclose(
        HnswIndex(vectors).vectors, BruteForceIndex(vectors).vectors, atol=1e-6
    )


# Searching


def test_search_finds_the_query_itself(vectors: np.ndarray) -> None:
    """A gallery vector used as a query is its own nearest neighbour."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        _, ids = index.search(vectors[:5], k=3)
        assert np.array_equal(ids[:, 0], np.arange(5))


def test_hnsw_matches_the_exact_ranking(vectors: np.ndarray) -> None:
    """On a small gallery the HNSW graph reproduces the exact ranking."""
    scores, ids = HnswIndex(vectors).search(vectors[:5], k=5)
    exact_scores, exact_ids = BruteForceIndex(vectors).search(vectors[:5], k=5)
    assert np.array_equal(ids, exact_ids)
    assert np.allclose(scores, exact_scores, atol=1e-5)


def test_search_accepts_a_single_vector(vectors: np.ndarray) -> None:
    """A ``(D,)`` query is searched as a batch of one."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        scores, ids = index.search(vectors[0], k=4)
        assert scores.shape == (1, 4)
        assert ids[0, 0] == 0


def test_search_pads_a_gallery_smaller_than_k(vectors: np.ndarray) -> None:
    """Missing neighbours are reported as the id ``-1``."""
    for index in (HnswIndex(vectors[:3]), BruteForceIndex(vectors[:3])):
        scores, ids = index.search(vectors[:1], k=5)
        assert ids.shape == (1, 5)
        assert list(ids[0, 3:]) == [-1, -1]
        assert np.isinf(scores[0, 3:]).all()


def test_hnsw_widens_its_walk_for_a_large_k(vectors: np.ndarray) -> None:
    """A search asking for more neighbours than ``search_candidates`` returns k."""
    index = HnswIndex(vectors, search_candidates=2)
    scores, ids = index.search(vectors[:1], k=20)
    assert ids.shape == (1, 20)
    assert (ids >= 0).all()
    assert index.search_candidates >= 20


def test_scores_rank_the_closest_first(vectors: np.ndarray) -> None:
    """Scores are distances, so they grow along a result row."""
    for index in (HnswIndex(vectors), BruteForceIndex(vectors)):
        scores, _ = index.search(vectors[:3], k=6)
        assert (np.diff(scores, axis=1) >= -1e-6).all()


# Rejected input


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_unknown_space_raises(vectors: np.ndarray, index_cls: type) -> None:
    """An unsupported metric space is rejected."""
    with pytest.raises(ValueError, match="Unsupported space"):
        index_cls(vectors, space="manhattan")


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_empty_gallery_raises(index_cls: type) -> None:
    """An index cannot be built over an empty gallery."""
    with pytest.raises(ValueError, match="empty gallery"):
        index_cls(np.zeros((0, 4), dtype=np.float32))


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_non_matrix_gallery_raises(index_cls: type) -> None:
    """A gallery that is not a 2-D matrix is rejected."""
    with pytest.raises(ValueError, match="2-D"):
        index_cls(np.zeros((2, 2, 2), dtype=np.float32))


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_non_positive_k_raises(vectors: np.ndarray, index_cls: type) -> None:
    """``k`` must be a positive integer."""
    index = index_cls(vectors)
    with pytest.raises(ValueError, match="positive integer"):
        index.search(vectors[:1], k=0)


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_query_with_wrong_dimensionality_raises(
    vectors: np.ndarray, index_cls: type
) -> None:
    """A query whose width differs from the indexed one is rejected."""
    index = index_cls(vectors)
    with pytest.raises(ValueError, match="dimensionality"):
        index.search(np.zeros((1, 3), dtype=np.float32), k=2)


# Rebuilding


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_update_replaces_the_gallery(vectors: np.ndarray, index_cls: type) -> None:
    """``update`` rebuilds the index over a new gallery."""
    index = index_cls(vectors)
    index.update(vectors[:10])
    assert len(index) == 10
    assert index.vectors.shape == (10, vectors.shape[1])
    _, ids = index.search(vectors[:1], k=3)
    assert ids[0, 0] == 0
    assert (ids < 10).all()


@pytest.mark.parametrize("index_cls", [HnswIndex, BruteForceIndex])
def test_update_keeps_the_dimensionality(vectors: np.ndarray, index_cls: type) -> None:
    """``update`` rejects a gallery of a different dimensionality."""
    index = index_cls(vectors)
    with pytest.raises(ValueError, match="dimensionality"):
        index.update(np.zeros((5, 3), dtype=np.float32))


def test_update_keeps_the_hnsw_parameters(vectors: np.ndarray) -> None:
    """The rebuilt graph keeps the parameters it was configured with."""
    index = HnswIndex(
        vectors, graph_degree=8, build_candidates=64, search_candidates=17
    )
    index.update(vectors[:10])
    assert index.graph_degree == 8
    assert index.build_candidates == 64
    assert index.search_candidates == 17
    assert index.max_elements == 10
