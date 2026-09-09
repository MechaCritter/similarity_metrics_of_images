"""Tests for :class:`pyvisim.image_store.ExternalSearchIndex`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyvisim.image_store import ExternalSearchIndex

faiss = pytest.importorskip("faiss")


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    """A ``(30, 8)`` gallery matrix, L2-normalised for inner-product search.

    :returns: a float32 matrix ready to be added to a FAISS index.
    """
    rng = np.random.default_rng(0)
    matrix = rng.random((30, 8), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix)


@pytest.fixture
def flat_index(vectors: np.ndarray) -> Any:
    """A flat inner-product FAISS index over the gallery.

    :param vectors: the gallery matrix.
    :returns: a populated ``faiss.IndexFlatIP``.
    """
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def test_reconstructs_the_vectors_of_a_flat_index(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """A flat index hands its vectors back, so none need to be passed."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    assert index.vectors.shape == vectors.shape
    assert np.allclose(index.vectors, vectors, atol=1e-6)
    assert len(index) == vectors.shape[0]
    assert index.dim == vectors.shape[1]


def test_defaults_to_the_external_name(flat_index: Any) -> None:
    """An unnamed index reports the default name."""
    assert ExternalSearchIndex.from_faiss_index(flat_index).name == "external"


def test_keeps_the_given_name(flat_index: Any) -> None:
    """A named index reports the name it was given."""
    index = ExternalSearchIndex.from_faiss_index(flat_index, name="flat-ip")
    assert index.name == "flat-ip"
    assert "flat-ip" in repr(index)


def test_wrapped_index_is_reachable(flat_index: Any) -> None:
    """The wrapped index stays available for its own API."""
    assert ExternalSearchIndex.from_faiss_index(flat_index).index is flat_index


def test_vectors_are_read_only(flat_index: Any) -> None:
    """The gallery an external index hands out cannot be written to."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    with pytest.raises(ValueError, match="read-only"):
        index.vectors[0, 0] = 1.0


def test_vectors_at_reads_the_requested_rows(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """``vectors_at`` hands back the named rows of the gallery, read-only."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    block = index.vectors_at([4, 1])
    assert np.allclose(block, vectors[[4, 1]], atol=1e-6)
    assert not block.flags.writeable
    for bad_ids in ([], [vectors.shape[0]]):
        with pytest.raises(ValueError, match="'ids'"):
            index.vectors_at(bad_ids)


def test_search_forwards_to_the_wrapped_index(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """Queries and their scores pass straight through."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    scores, ids = index.search(vectors[:3], k=4)
    assert scores.shape == (3, 4)
    assert ids.shape == (3, 4)
    assert np.array_equal(ids[:, 0], np.arange(3))
    # An inner-product index scores similarity, so a self-match scores 1.
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)


def test_search_accepts_a_single_vector(flat_index: Any, vectors: np.ndarray) -> None:
    """A ``(D,)`` query is searched as a batch of one."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    scores, ids = index.search(vectors[0], k=2)
    assert scores.shape == (1, 2)
    assert ids[0, 0] == 0


def test_search_rejects_a_wrong_dimensionality(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """A query whose width differs from the indexed one is rejected."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    with pytest.raises(ValueError, match="dimensionality"):
        index.search(np.zeros((1, 3), dtype=np.float32), k=2)


def test_search_rejects_a_non_positive_k(flat_index: Any, vectors: np.ndarray) -> None:
    """``k`` must be a positive integer."""
    index = ExternalSearchIndex.from_faiss_index(flat_index)
    with pytest.raises(ValueError, match="positive integer"):
        index.search(vectors[:1], k=0)


def test_reconstructs_the_vectors_of_an_ivf_index(vectors: np.ndarray) -> None:
    """An IVF-Flat index hands its vectors back as they were added."""
    quantizer = faiss.IndexFlatL2(vectors.shape[1])
    ivf = faiss.IndexIVFFlat(quantizer, vectors.shape[1], 4)
    ivf.train(vectors)
    ivf.add(vectors)

    index = ExternalSearchIndex.from_faiss_index(ivf)
    assert np.allclose(index.vectors, vectors, atol=1e-6)


def test_index_that_cannot_reconstruct_needs_explicit_vectors(
    vectors: np.ndarray,
) -> None:
    """An index with nothing to reconstruct from asks for its vectors."""

    class _NoReconstruct:
        """A stand-in that can search but cannot give its vectors back."""

        ntotal = 30

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Return an empty neighbour block for every query.

            :param queries: the ``(M, D)`` query batch.
            :param k: number of neighbours per query.
            :returns: a ``(scores, ids)`` pair of ``(M, k)`` arrays.
            """
            rows = queries.shape[0]
            return np.zeros((rows, k), np.float32), np.full((rows, k), -1)

    with pytest.raises(ValueError, match="must be passed explicitly"):
        ExternalSearchIndex.from_faiss_index(_NoReconstruct())

    index = ExternalSearchIndex.from_faiss_index(_NoReconstruct(), vectors)
    assert np.allclose(index.vectors, vectors, atol=1e-6)


def test_compressed_index_reconstructs_an_approximation(vectors: np.ndarray) -> None:
    """A product-quantized index gives back a lossy version of its gallery."""
    pq = faiss.IndexPQ(vectors.shape[1], 4, 4)
    pq.train(vectors)
    pq.add(vectors)

    index = ExternalSearchIndex.from_faiss_index(pq)
    assert index.vectors.shape == vectors.shape
    assert not np.allclose(index.vectors, vectors, atol=1e-6)


def test_explicit_vectors_win_over_reconstruction(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """Passed vectors are kept as they are, without a reconstruction."""
    passed = vectors * 2.0
    index = ExternalSearchIndex.from_faiss_index(flat_index, passed)
    assert np.allclose(index.vectors, passed, atol=1e-6)


def test_rejects_a_size_mismatch(flat_index: Any, vectors: np.ndarray) -> None:
    """Vectors that do not cover the whole index are rejected."""
    with pytest.raises(ValueError, match="vectors, but"):
        ExternalSearchIndex.from_faiss_index(flat_index, vectors[:5])


def test_rejects_an_index_without_search(vectors: np.ndarray) -> None:
    """An object with no ``search`` method cannot be adapted."""
    with pytest.raises(AttributeError, match="no 'search' method"):
        ExternalSearchIndex(object(), vectors)


def test_wraps_any_object_that_can_search(vectors: np.ndarray) -> None:
    """FAISS is not required: anything with a ``search`` method works."""

    class _Stub:
        """A minimal stand-in returning the first ``k`` gallery rows."""

        ntotal = 30

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Return a fixed neighbour block for every query.

            :param queries: the ``(M, D)`` query batch.
            :param k: number of neighbours per query.
            :returns: a ``(scores, ids)`` pair of ``(M, k)`` arrays.
            """
            rows = queries.shape[0]
            return (
                np.zeros((rows, k), dtype=np.float32),
                np.tile(np.arange(k), (rows, 1)),
            )

    index = ExternalSearchIndex(_Stub(), vectors, name="stub")
    scores, ids = index.search(vectors[:2], k=3)
    assert scores.shape == (2, 3)
    assert np.array_equal(ids, np.tile(np.arange(3), (2, 1)))
