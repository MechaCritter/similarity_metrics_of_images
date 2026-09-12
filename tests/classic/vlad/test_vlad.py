"""Tests for :class:`pyvisim.classic.VLADEmbedder`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim._errors import NotFittedError
from pyvisim.classic import VLADEmbedder
from pyvisim.classic._clustering import DiagCovarGaussianMixture, KMeans
from pyvisim.serialization import EMBEDDER_METADATA_KEY, save_state

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import ImageObj

#: VLAD output width for ``n_clusters=8`` and RootSIFT (128-d): ``8 * 128``.
VLAD_DIM_NO_PCA = 8 * 128
#: VLAD output width with PCA to 32 dimensions: ``8 * 32``.
VLAD_DIM_PCA = 8 * 32

BATCH_SIZES = [2, 3, 4, 6, 16]


@pytest.fixture(scope="module")
def vlad_no_pca(category_train_images_flat: list[np.ndarray]) -> VLADEmbedder:
    """A learned ``VLADEmbedder(n_clusters=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEmbedder``.
    """
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    embedder.learn(category_train_images_flat)
    return embedder


@pytest.fixture(scope="module")
def vlad_pca(category_train_images_flat: list[np.ndarray]) -> VLADEmbedder:
    """A learned ``VLADEmbedder(n_clusters=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEmbedder``.
    """
    embedder = VLADEmbedder(
        n_clusters=8,
        kmeans_params={"rng": 0},
        pca_params={"n_components": 32, "rng": 0},
    )
    embedder.learn(category_train_images_flat)
    return embedder


def test_clustering_model_type() -> None:
    """VLAD builds a KMeans clustering model."""
    assert isinstance(VLADEmbedder(n_clusters=8).clustering_model, KMeans)


def test_n_clusters_param_kwargs_collision() -> None:
    """Passing ``n_clusters`` inside ``kmeans_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_clusters' directly"):
        VLADEmbedder(kmeans_params={"n_clusters": 8})


def test_clustering_model_is_read_only() -> None:
    """``clustering_model`` is a read-only property; direct assignment raises ``AttributeError``."""
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(AttributeError):
        embedder.clustering_model = DiagCovarGaussianMixture(8)  # type: ignore[assignment]


def test_embed_shape_no_pca(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image embeds to ``(1, n_clusters * 128)``."""
    out = vlad_no_pca.embed([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_NO_PCA)


def test_embed_batch_shape(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A batch of three images embeds to ``(3, n_clusters * 128)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0), np.roll(base, 8, axis=1)]
    assert vlad_no_pca.embed(batch).shape == (3, VLAD_DIM_NO_PCA)


def test_embed_shape_with_pca(
    vlad_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image embeds to ``(1, n_clusters * 32)``."""
    out = vlad_pca.embed([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_PCA)


def test_cluster_rows_l2_normalized(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """Each non-zero cluster row of the encoding has unit L2 norm."""
    raw = VLADEmbedder.from_dict({**vlad_no_pca.to_dict(), "normalize": False})
    out = raw.embed([checkerboard_image.array]).reshape(8, 128)
    norms = np.linalg.norm(out, axis=1)
    non_zero = norms[norms > 1e-6]
    assert non_zero == pytest.approx(np.ones_like(non_zero), rel=1e-3)


def test_embed_rows_l2_normalized(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """``embed`` returns unit-length rows while ``normalize`` is on."""
    out = vlad_no_pca.embed([checkerboard_image.array])
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(len(out)), rel=1e-5)


def test_embed_single_rgb_image_ok(
    vlad_no_pca: VLADEmbedder, rgb_image: ImageObj
) -> None:
    """A bare 3-D RGB array is treated as a single image."""
    assert vlad_no_pca.embed(rgb_image.array).shape == (1, VLAD_DIM_NO_PCA)


def test_embed_single_2d_must_be_wrapped(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A single 2-D image must be wrapped in a list; a bare 2-D array iterates rows."""
    gray = checkerboard_image.array
    assert vlad_no_pca.embed([gray]).shape == (1, VLAD_DIM_NO_PCA)
    with pytest.raises(ValueError):
        vlad_no_pca.embed(gray)


def test_embed_no_descriptors_raises(
    vlad_no_pca: VLADEmbedder, solid_image: ImageObj
) -> None:
    """Embedding a featureless image raises a clear ``ValueError``."""
    with pytest.raises(ValueError, match="No descriptors found in the image"):
        vlad_no_pca.embed([solid_image.array])


def test_embed_accepts_tensor(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert vlad_no_pca.embed([tensor]).shape == (1, VLAD_DIM_NO_PCA)


def test_predict_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Embedding before learning raises ``NotFittedError`` (KMeans not fitted)."""
    with pytest.raises(NotFittedError):
        VLADEmbedder(n_clusters=8).embed([checkerboard_image.array])


# Batching


@pytest.fixture(scope="module")
def sample_images(category_train_images_flat: list[np.ndarray]) -> list[np.ndarray]:
    """Four training images, enough for every batch size to split them.

    :param category_train_images_flat: flattened training images.
    :returns: a short list of images.
    """
    return category_train_images_flat[:4]


@pytest.fixture
def batched_vlad(vlad_no_pca: VLADEmbedder) -> Iterator[VLADEmbedder]:
    """The learned embedder, with its batch size restored after the test.

    :param vlad_no_pca: the learned embedder shared by this module.
    :returns: the same embedder, ready to be rebatched.
    """
    original = vlad_no_pca.batch_size
    yield vlad_no_pca
    vlad_no_pca.set_batch_size(original)


@pytest.fixture(scope="module")
def unbatched_results(
    vlad_no_pca: VLADEmbedder, sample_images: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The embeddings and scores of the sample images, one image per batch.

    :param vlad_no_pca: the learned embedder shared by this module.
    :param sample_images: the images to embed.
    :returns: the ``(embeddings, scores)`` every batch size has to reproduce.
    """
    original = vlad_no_pca.batch_size
    vlad_no_pca.set_batch_size(1)
    try:
        return (
            vlad_no_pca.embed(sample_images),
            vlad_no_pca.similarity_score(sample_images[:2], sample_images[2:]),
        )
    finally:
        vlad_no_pca.set_batch_size(original)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_the_embeddings(
    batched_vlad: VLADEmbedder,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the work: vectors and scores stay in place."""
    embeddings, scores = unbatched_results
    batched_vlad.set_batch_size(batch_size)
    np.testing.assert_allclose(
        batched_vlad.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_vlad.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_whole_input_as_one_batch_does_not_change_the_embeddings(
    batched_vlad: VLADEmbedder,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
) -> None:
    """``-1`` embeds the input in one go, for the same vectors as ``1`` does."""
    embeddings, scores = unbatched_results
    batched_vlad.set_batch_size(-1)
    np.testing.assert_allclose(
        batched_vlad.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_vlad.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_embedding_image_by_image_matches_the_whole_input(
    batched_vlad: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """One call per image stacks into what one call over the input returns."""
    batched_vlad.set_batch_size(1)
    one_at_a_time = np.vstack([batched_vlad.embed([image]) for image in sample_images])
    np.testing.assert_allclose(
        batched_vlad.embed(sample_images), one_at_a_time, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(("n_images", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_images(
    batched_vlad: VLADEmbedder,
    category_train_images_flat: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    n_images: int,
    batch_size: int,
) -> None:
    """The last batch holds ``N % batch_size`` images, however small ``N`` is."""
    batch_lengths: list[int] = []
    embed_batch = batched_vlad._embed

    def record(images: list[np.ndarray]) -> np.ndarray:
        batch_lengths.append(len(images))
        return embed_batch(images)

    monkeypatch.setattr(batched_vlad, "_embed", record)
    batched_vlad.set_batch_size(batch_size)
    batched_vlad.embed(category_train_images_flat[:n_images])
    assert batch_lengths[-1] == n_images % batch_size
    assert sum(batch_lengths) == n_images


def test_embedding_matrix_shape_for_a_single_image(
    vlad_no_pca: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """A single image keeps its own row of the embedding matrix."""
    assert vlad_no_pca.embed(sample_images[:1]).shape == (1, VLAD_DIM_NO_PCA)


def test_embedding_nothing_raises(vlad_no_pca: VLADEmbedder) -> None:
    """An input holding no image is an error, not an empty embedding matrix."""
    with pytest.raises(ValueError):
        vlad_no_pca.embed([])


def test_the_batch_size_applies_to_a_generator_of_images(
    batched_vlad: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """A generator is consumed batch by batch and embeds like a list."""
    batched_vlad.set_batch_size(2)
    from_generator = batched_vlad.embed(image for image in sample_images)
    np.testing.assert_array_equal(from_generator, batched_vlad.embed(sample_images))


def test_unflattened_batching_keeps_the_layout(
    batched_vlad: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """With ``flatten`` off, batches stack per-cluster rows just as singles do."""
    batched_vlad.flatten = False
    try:
        batched_vlad.set_batch_size(1)
        one_by_one = batched_vlad.embed(sample_images)
        batched_vlad.set_batch_size(3)
        batched = batched_vlad.embed(sample_images)
    finally:
        batched_vlad.flatten = True
    n_clusters = batched_vlad.clustering_model.n_clusters
    assert one_by_one.shape[0] == len(sample_images) * n_clusters
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-5, atol=1e-6)


def test_a_featureless_image_is_reported_inside_a_batch(
    batched_vlad: VLADEmbedder,
    sample_images: list[np.ndarray],
    solid_image: ImageObj,
) -> None:
    """An image yielding no descriptor is still caught when it shares a batch."""
    batched_vlad.set_batch_size(-1)
    with pytest.raises(ValueError, match="No descriptors found in the image"):
        batched_vlad.embed([*sample_images, solid_image.array])


@pytest.mark.parametrize("batch_size", [2, -1])
def test_learning_does_not_depend_on_the_batch_size(
    sample_images: list[np.ndarray], batch_size: int
) -> None:
    """The vocabulary is fitted on the same features however they are batched."""
    reference = VLADEmbedder(n_clusters=4, kmeans_params={"rng": 0})
    reference.learn(sample_images)
    embedder = VLADEmbedder(
        n_clusters=4, kmeans_params={"rng": 0}, batch_size=batch_size
    )
    embedder.learn(sample_images)
    np.testing.assert_allclose(
        embedder.clustering_model.cluster_centers,
        reference.clustering_model.cluster_centers,
        rtol=1e-5,
        atol=1e-6,
    )


def test_batch_size_survives_a_file_round_trip(
    batched_vlad: VLADEmbedder, tmp_path: Path
) -> None:
    """A saved embedder comes back with the batch size it was saved with."""
    batched_vlad.set_batch_size(8)
    path = batched_vlad.save_to_disk(tmp_path / "vlad")
    assert VLADEmbedder.load_from_disk(path).batch_size == 8


def test_a_state_without_a_batch_size_is_not_a_valid_file(
    vlad_no_pca: VLADEmbedder, tmp_path: Path
) -> None:
    """The batch size is a required key, so a file predating it is rejected."""
    state = vlad_no_pca.to_dict()
    del state["batch_size"]
    save_state(state, path := tmp_path / "vlad.embedder", EMBEDDER_METADATA_KEY)
    with pytest.raises(ValueError, match="not a valid .embedder file"):
        VLADEmbedder.load_from_disk(path)
