"""Tests for :class:`pyvisim.image_store.KReciprocalReranker`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from pyvisim.classic import VLADEmbedder
from pyvisim.image_store import (
    Candidate,
    ExternalSearchIndex,
    InMemoryImageEmbeddingStore,
    KReciprocalReranker,
)

# The distance computation is checked on its own against a literal transcription
# of the paper: the reranker only ever feeds it matrices it assembled itself.
from pyvisim.image_store.reranking import _k_reciprocal_distances


@pytest.fixture(scope="module")
def gallery(
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images: dict[str, list[np.ndarray]],
) -> tuple[list[str], dict[str, set[str]]]:
    """Write the training images to disk, grouped by category.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :param category_train_images: per-category training images.
    :returns: every gallery path, and the paths of each category.
    """
    directory = tmp_path_factory.mktemp("rerank_gallery")
    paths: list[str] = []
    by_category: dict[str, set[str]] = {}
    for name, images in category_train_images.items():
        for index, image in enumerate(images):
            path = directory / f"{name}_{index}.png"
            Image.fromarray(np.stack([image, image, image], axis=-1)).save(path)
            paths.append(str(path))
            by_category.setdefault(name, set()).add(str(path))
    return paths, by_category


@pytest.fixture(scope="module")
def store(
    gallery: tuple[list[str], dict[str, set[str]]],
    learned_vlad_embedder: VLADEmbedder,
) -> InMemoryImageEmbeddingStore:
    """A brute-force store over the two-category gallery.

    :param gallery: the gallery paths and their categories.
    :param learned_vlad_embedder: a fitted VLAD embedder.
    :returns: a store backed by an exact index.
    """
    paths, _ = gallery
    return InMemoryImageEmbeddingStore(paths, learned_vlad_embedder)


@pytest.fixture(scope="module")
def reranker(store: InMemoryImageEmbeddingStore) -> KReciprocalReranker:
    """A reranker with neighbourhoods sized for the 20-image gallery.

    :param store: the store the candidates come from.
    :returns: a reranker with ``k1=6``, ``k2=3`` and the paper's ``lambda``.
    """
    return KReciprocalReranker(store, k1=6, k2=3, lambda_value=0.3)


def _probe(gray: np.ndarray) -> np.ndarray:
    """Stack a grayscale image into the RGB array the store embeds.

    :param gray: a ``(H, W)`` image.
    :returns: the ``(H, W, 3)`` image.
    """
    return np.stack([gray, gray, gray], axis=-1)


def _random_distances(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw a symmetric distance matrix with a zero diagonal.

    :param rng: the random generator.
    :param size: number of elements, the probe included.
    :returns: a ``(size, size)`` matrix.
    """
    upper = np.triu(rng.random((size, size)), k=1)
    return upper + upper.T


def _paper_distances(
    distances: np.ndarray, k1: int, k2: int, lambda_value: float
) -> np.ndarray:
    """Transcribe Section 3 of Zhong et al. literally, with Python sets.

    Every equation of the paper is spelled out over the set of the probe (index
    ``0``) and the candidates, with the conventions of the authors' code: the
    ranking lists include the element itself, the original distances are
    scaled by their row maximum and the features are L1-normalised.

    :param distances: the symmetric pairwise distance matrix, probe first.
    :param k1: neighbourhood size of the k-reciprocal sets.
    :param k2: neighbourhood size of the local query expansion.
    :param lambda_value: weight of the original distance.
    :returns: the final distances from the probe to every candidate.
    """
    scaled = distances / distances.max(axis=1, keepdims=True)
    size = scaled.shape[0]
    ranking = [list(np.argsort(scaled[i], kind="stable")) for i in range(size)]

    def neighbours(i: int, k: int) -> set[int]:
        """Eq. (2): the k nearest neighbours, the element itself included."""
        return set(ranking[i][: k + 1])

    def reciprocal(i: int, k: int) -> set[int]:
        """Eq. (3): the neighbours that hold ``i`` among their own."""
        return {g for g in neighbours(i, k) if i in neighbours(g, k)}

    half = int(np.around(k1 / 2))
    expanded: list[set[int]] = []
    for p in range(size):
        r_p = reciprocal(p, k1)
        r_star = set(r_p)
        for q in r_p:  # Eq. (4)
            r_q = reciprocal(q, half)
            if len(r_p & r_q) >= 2 / 3 * len(r_q):
                r_star |= r_q
        expanded.append(r_star)

    features = np.zeros((size, size))
    for p in range(size):  # Eq. (7)
        for g in expanded[p]:
            features[p, g] = np.exp(-scaled[p, g])
        features[p] /= features[p].sum()
    if k2 > 1:  # Eq. (11)
        features = np.stack(
            [
                np.mean([features[g] for g in ranking[p][:k2]], axis=0)
                for p in range(size)
            ]
        )

    final = []
    for g in range(1, size):  # Eq. (10) and (12)
        intersection = np.minimum(features[0], features[g]).sum()
        union = np.maximum(features[0], features[g]).sum()
        jaccard = 1.0 - intersection / union
        final.append((1.0 - lambda_value) * jaccard + lambda_value * scaled[0, g])
    return np.array(final)


# Construction


def test_exposes_its_parameters(store: InMemoryImageEmbeddingStore) -> None:
    """The reranker reports the store and the parameters it was built with."""
    reranker = KReciprocalReranker(store, k1=10, k2=4, lambda_value=0.5)
    assert reranker.store is store
    assert reranker.k1 == 10
    assert reranker.k2 == 4
    assert reranker.lambda_value == 0.5
    assert "k1=10" in repr(reranker)


def test_defaults_follow_the_paper(store: InMemoryImageEmbeddingStore) -> None:
    """The defaults are the values Zhong et al. use."""
    reranker = KReciprocalReranker(store)
    assert (reranker.k1, reranker.k2, reranker.lambda_value) == (20, 6, 0.3)


def test_rejects_a_non_store() -> None:
    """Anything but an ``InMemoryImageEmbeddingStore`` is refused."""
    with pytest.raises(TypeError, match="InMemoryImageEmbeddingStore"):
        KReciprocalReranker(object())  # type: ignore[arg-type]


def test_rejects_a_store_on_an_external_index(
    store: InMemoryImageEmbeddingStore,
) -> None:
    """A store searching through an external index has scores of unknown meaning."""

    class _Stub:
        """A stand-in index that can search but says nothing about its metric."""

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Return the first ``k`` gallery rows for every query.

            :param queries: the ``(M, D)`` query batch.
            :param k: number of neighbours per query.
            :returns: a ``(scores, ids)`` pair of ``(M, k)`` arrays.
            """
            rows = queries.shape[0]
            return np.zeros((rows, k), np.float32), np.tile(np.arange(k), (rows, 1))

    external = InMemoryImageEmbeddingStore(
        store.paths, store.embedder, ExternalSearchIndex(_Stub(), store.embeddings)
    )
    with pytest.raises(ValueError, match="external index"):
        KReciprocalReranker(external)


@pytest.mark.parametrize(
    "params",
    [
        {"k1": 0},
        {"k1": 2.5},
        {"k2": 0},
        {"k1": 5, "k2": 6},
        {"lambda_value": -0.1},
        {"lambda_value": 1.5},
        {"lambda_value": "0.3"},
    ],
)
def test_rejects_bad_parameters(
    store: InMemoryImageEmbeddingStore, params: dict[str, Any]
) -> None:
    """The neighbourhood sizes and the mixing weight are checked up front."""
    with pytest.raises(ValueError):
        KReciprocalReranker(store, **params)


# Rejected candidate lists


def test_rejects_a_non_positive_top_k(
    reranker: KReciprocalReranker, store: InMemoryImageEmbeddingStore
) -> None:
    """``top_k`` must be a positive integer."""
    candidates = [Candidate(store.paths[0], 0.1)]
    for top_k in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="'top_k'"):
            reranker.rerank(candidates, top_k)  # type: ignore[arg-type]


def test_rejects_no_candidates(reranker: KReciprocalReranker) -> None:
    """There must be at least one candidate to re-rank."""
    with pytest.raises(ValueError, match="at least one"):
        reranker.rerank([], top_k=1)


def test_rejects_a_non_candidate(
    reranker: KReciprocalReranker, store: InMemoryImageEmbeddingStore
) -> None:
    """A bare ``(path, score)`` tuple is not a candidate."""
    with pytest.raises(TypeError, match="Candidate"):
        reranker.rerank([(store.paths[0], 0.1)], top_k=1)  # type: ignore[list-item]


def test_rejects_a_path_outside_the_store(reranker: KReciprocalReranker) -> None:
    """A candidate the store cannot look up is reported."""
    with pytest.raises(ValueError, match="not in the gallery"):
        reranker.rerank([Candidate("absent.png", 0.1)], top_k=1)


def test_rejects_a_duplicated_path(
    reranker: KReciprocalReranker, store: InMemoryImageEmbeddingStore
) -> None:
    """A path may appear once among the candidates."""
    twice = [Candidate(store.paths[0], 0.1), Candidate(store.paths[0], 0.2)]
    with pytest.raises(ValueError, match="twice"):
        reranker.rerank(twice, top_k=1)


def test_rejects_a_non_finite_score(
    reranker: KReciprocalReranker, store: InMemoryImageEmbeddingStore
) -> None:
    """A padded or broken score cannot serve as a distance."""
    with pytest.raises(ValueError, match="finite"):
        reranker.rerank([Candidate(store.paths[0], float("inf"))], top_k=1)


# Re-ranking


def test_returns_the_best_top_k(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """The result holds ``top_k`` distinct input candidates, best first."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[0]), k=len(store)
    )[0]
    best = reranker.rerank(candidates, top_k=5)
    assert len(best) == 5
    assert all(isinstance(candidate, Candidate) for candidate in best)
    assert {candidate.path for candidate in best} <= {c.path for c in candidates}
    assert len({candidate.path for candidate in best}) == 5
    scores = [candidate.score for candidate in best]
    assert scores == sorted(scores)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_top_k_beyond_the_pool_returns_every_candidate(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Asking for more than there are candidates hands all of them back."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[0]), k=4
    )[0]
    assert len(reranker.rerank(candidates, top_k=50)) == 4


def test_a_single_candidate_survives(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A pool of one is capped to and returned as itself."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[0]), k=1
    )[0]
    best = reranker.rerank(candidates, top_k=3)
    assert [candidate.path for candidate in best] == [candidates[0].path]


def test_keeps_the_query_image_first(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A gallery image used as the query still comes out on top."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[3]), k=len(store)
    )[0]
    assert reranker.rerank(candidates, top_k=1)[0].path == store.paths[3]


def test_keeps_the_query_category_on_top(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    gallery: tuple[list[str], dict[str, set[str]]],
    category_query_images: dict[str, list[np.ndarray]],
) -> None:
    """After re-ranking, the best matches of a query share its category."""
    _, by_category = gallery
    for name, images in category_query_images.items():
        candidates = store.retrieve_top_k_similar(_probe(images[0]), k=len(store))[0]
        best = reranker.rerank(candidates, top_k=5)
        assert {candidate.path for candidate in best} <= by_category[name]


def test_demotes_an_impostor_with_a_foreign_neighbourhood(
    store: InMemoryImageEmbeddingStore,
    gallery: tuple[list[str], dict[str, set[str]]],
    category_query_images: dict[str, list[np.ndarray]],
) -> None:
    """A perfect score does not save a candidate whose neighbours disagree.

    The best match of the other category is planted at the top of the ranking
    with a distance of zero. The Jaccard distance alone (``lambda_value=0``)
    then ranks it below the query's own category, whose neighbourhoods overlap
    with the query's while the impostor's does not.
    """
    _, by_category = gallery
    first, second = list(by_category)
    candidates = store.retrieve_top_k_similar(
        _probe(category_query_images[first][0]), k=len(store)
    )[0]
    impostor = next(c for c in candidates if c.path in by_category[second])
    planted = [Candidate(impostor.path, 0.0)] + [
        c for c in candidates if c.path != impostor.path
    ]
    reranker = KReciprocalReranker(store, k1=6, k2=1, lambda_value=0.0)
    best = reranker.rerank(planted, top_k=len(planted))
    order = [candidate.path for candidate in best]
    assert order[0] in by_category[first]
    assert order.index(impostor.path) > 0


def test_lambda_one_keeps_the_original_order(
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """With the Jaccard distance weighted out, the store's ranking is kept."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[1]), k=len(store)
    )[0]
    kept = KReciprocalReranker(store, k1=6, k2=3, lambda_value=1.0)
    best = kept.rerank(candidates, top_k=len(candidates))
    assert [c.path for c in best] == [c.path for c in candidates]


def test_original_scores_are_left_untouched(
    reranker: KReciprocalReranker,
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Re-ranking returns new candidates instead of rewriting the given ones."""
    candidates = store.retrieve_top_k_similar(
        _probe(category_train_images_flat[0]), k=len(store)
    )[0]
    before = [(c.path, c.score) for c in candidates]
    reranker.rerank(candidates, top_k=3)
    assert [(c.path, c.score) for c in candidates] == before


# The distance computation against the paper


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ("size", "k1", "k2", "lambda_value"),
    [
        (8, 3, 2, 0.3),
        (12, 5, 3, 0.3),
        (25, 20, 6, 0.3),
        (25, 10, 1, 0.0),
        (30, 7, 4, 1.0),
    ],
)
def test_distances_match_a_literal_transcription_of_the_paper(
    seed: int, size: int, k1: int, k2: int, lambda_value: float
) -> None:
    """The vectorised computation agrees with the equations spelled out in sets."""
    distances = _random_distances(np.random.default_rng(seed), size)
    ours = _k_reciprocal_distances(distances, k1, k2, lambda_value)
    reference = _paper_distances(distances, k1, k2, lambda_value)
    assert ours.shape == (size - 1,)
    assert np.allclose(ours, reference, atol=1e-12)


def test_distances_lie_in_the_unit_interval() -> None:
    """Both the Jaccard and the scaled original distance lie in ``[0, 1]``."""
    distances = _random_distances(np.random.default_rng(42), 20) * 100.0
    for lambda_value in (0.0, 0.5, 1.0):
        final = _k_reciprocal_distances(distances, 8, 4, lambda_value)
        assert (final >= 0.0).all()
        assert (final <= 1.0).all()


def test_a_duplicate_of_the_probe_gets_distance_zero() -> None:
    """A candidate at distance zero from the probe shares its neighbourhood."""
    distances = _random_distances(np.random.default_rng(7), 12)
    distances[1, :] = distances[0, :]
    distances[:, 1] = distances[:, 0]
    distances[0, 1] = distances[1, 0] = 0.0
    final = _k_reciprocal_distances(distances, 5, 3, 0.3)
    assert final[0] == pytest.approx(0.0, abs=1e-12)
    assert final[0] <= final.min()
