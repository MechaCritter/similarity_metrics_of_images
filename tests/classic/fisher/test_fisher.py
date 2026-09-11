"""Tests for :class:`pyvisim.classic.FisherVectorEmbedder`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim._errors import NotFittedError
from pyvisim.classic import FisherVectorEmbedder
from pyvisim.classic._clustering import DiagCovarGaussianMixture, KMeans

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import ImageObj

#: Fisher output width for ``k=8`` and RootSIFT (128-d): ``2 * 8 * 128 + 8``.
FISHER_DIM_NO_PCA = 2 * 8 * 128 + 8
#: Fisher output width with PCA to 32 dimensions: ``2 * 8 * 32 + 8``.
FISHER_DIM_PCA = 2 * 8 * 32 + 8

BATCH_SIZES = [2, 3, 4, 6, 16]


@pytest.fixture(scope="module")
def fisher_no_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEmbedder:
    """A learned ``FisherVectorEmbedder(n_components=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEmbedder``.
    """
    embedder = FisherVectorEmbedder(n_components=8, gmm_params={"rng": 0})
    embedder.learn(category_train_images_flat)
    return embedder


@pytest.fixture(scope="module")
def fisher_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEmbedder:
    """A learned ``FisherVectorEmbedder(n_components=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEmbedder``.
    """
    embedder = FisherVectorEmbedder(
        n_components=8,
        gmm_params={"rng": 0},
        pca_params={"n_components": 32, "rng": 0},
    )
    embedder.learn(category_train_images_flat)
    return embedder


def test_clustering_model_type() -> None:
    """Fisher builds a Gaussian mixture clustering model."""
    model = FisherVectorEmbedder(n_components=8).clustering_model
    assert isinstance(model, DiagCovarGaussianMixture)


def test_n_components_kwargs_collision() -> None:
    """Passing ``n_components`` inside ``gmm_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_components' directly"):
        FisherVectorEmbedder(gmm_params={"n_components": 8})


def test_clustering_model_is_read_only() -> None:
    """``clustering_model`` is a read-only property; direct assignment raises ``AttributeError``."""
    embedder = FisherVectorEmbedder(n_components=8)
    with pytest.raises(AttributeError):
        embedder.clustering_model = KMeans(8)  # type: ignore[assignment]


def test_embed_shape_no_pca(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image embeds to ``(1, 2 * k * 128 + k)``."""
    out = fisher_no_pca.embed([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_NO_PCA)


def test_embed_shape_with_pca(
    fisher_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image embeds to ``(1, 2 * k * 32 + k)``."""
    out = fisher_pca.embed([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_PCA)


def test_embed_batch_shape(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """A batch of two images embeds to ``(2, 2 * k * 128 + k)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert fisher_no_pca.embed(batch).shape == (2, FISHER_DIM_NO_PCA)


def test_embed_accepts_tensor(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert fisher_no_pca.embed([tensor]).shape == (1, FISHER_DIM_NO_PCA)


def test_embed_no_descriptors(
    fisher_no_pca: FisherVectorEmbedder, solid_image: ImageObj
) -> None:
    """Embedding a featureless image raises (GMM cannot score zero descriptors)."""
    with pytest.raises(ValueError):
        fisher_no_pca.embed([solid_image.array])


def test_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Embedding before learning raises ``NotFittedError`` (GMM not fitted)."""
    with pytest.raises(NotFittedError):
        FisherVectorEmbedder(n_components=8).embed([checkerboard_image.array])


# Batching


@pytest.fixture(scope="module")
def sample_images(category_train_images_flat: list[np.ndarray]) -> list[np.ndarray]:
    """Four training images, enough for every batch size to split them.

    :param category_train_images_flat: flattened training images.
    :returns: a short list of images.
    """
    return category_train_images_flat[:4]


@pytest.fixture
def batched_fisher(
    fisher_no_pca: FisherVectorEmbedder,
) -> Iterator[FisherVectorEmbedder]:
    """The learned embedder, with its batch size restored after the test.

    :param fisher_no_pca: the learned embedder shared by this module.
    :returns: the same embedder, ready to be rebatched.
    """
    original = fisher_no_pca.batch_size
    yield fisher_no_pca
    fisher_no_pca.set_batch_size(original)


@pytest.fixture(scope="module")
def unbatched_results(
    fisher_no_pca: FisherVectorEmbedder, sample_images: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The embeddings and scores of the sample images, one image per batch.

    :param fisher_no_pca: the learned embedder shared by this module.
    :param sample_images: the images to embed.
    :returns: the ``(embeddings, scores)`` every batch size has to reproduce.
    """
    original = fisher_no_pca.batch_size
    fisher_no_pca.set_batch_size(1)
    try:
        return (
            fisher_no_pca.embed(sample_images),
            fisher_no_pca.similarity_score(sample_images[:2], sample_images[2:]),
        )
    finally:
        fisher_no_pca.set_batch_size(original)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_the_embeddings(
    batched_fisher: FisherVectorEmbedder,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the work: vectors and scores stay in place."""
    embeddings, scores = unbatched_results
    batched_fisher.set_batch_size(batch_size)
    np.testing.assert_allclose(
        batched_fisher.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_fisher.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_whole_input_as_one_batch_does_not_change_the_embeddings(
    batched_fisher: FisherVectorEmbedder,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
) -> None:
    """``-1`` embeds the input in one go, for the same vectors as ``1`` does."""
    embeddings, scores = unbatched_results
    batched_fisher.set_batch_size(-1)
    np.testing.assert_allclose(
        batched_fisher.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_fisher.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_embedding_image_by_image_matches_the_whole_input(
    batched_fisher: FisherVectorEmbedder, sample_images: list[np.ndarray]
) -> None:
    """One call per image stacks into what one call over the input returns."""
    batched_fisher.set_batch_size(1)
    one_at_a_time = np.vstack(
        [batched_fisher.embed([image]) for image in sample_images]
    )
    np.testing.assert_allclose(
        batched_fisher.embed(sample_images), one_at_a_time, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(("n_images", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_images(
    batched_fisher: FisherVectorEmbedder,
    category_train_images_flat: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    n_images: int,
    batch_size: int,
) -> None:
    """The last batch holds ``N % batch_size`` images, however small ``N`` is."""
    batch_lengths: list[int] = []
    embed_batch = batched_fisher._embed

    def record(images: list[np.ndarray]) -> np.ndarray:
        batch_lengths.append(len(images))
        return embed_batch(images)

    monkeypatch.setattr(batched_fisher, "_embed", record)
    batched_fisher.set_batch_size(batch_size)
    batched_fisher.embed(category_train_images_flat[:n_images])
    assert batch_lengths[-1] == n_images % batch_size
    assert sum(batch_lengths) == n_images


def test_embedding_matrix_shape_for_a_single_image(
    fisher_no_pca: FisherVectorEmbedder, sample_images: list[np.ndarray]
) -> None:
    """A single image keeps its own row of the embedding matrix."""
    assert fisher_no_pca.embed(sample_images[:1]).shape == (1, FISHER_DIM_NO_PCA)


def test_embedding_nothing_raises(fisher_no_pca: FisherVectorEmbedder) -> None:
    """An input holding no image is an error, not an empty embedding matrix."""
    with pytest.raises(ValueError):
        fisher_no_pca.embed([])


def test_batch_size_survives_a_file_round_trip(
    batched_fisher: FisherVectorEmbedder, tmp_path: Path
) -> None:
    """A saved embedder comes back with the batch size it was saved with."""
    batched_fisher.set_batch_size(8)
    path = batched_fisher.save_to_disk(tmp_path / "fisher")
    assert FisherVectorEmbedder.load_from_disk(path).batch_size == 8
