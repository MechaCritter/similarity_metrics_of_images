"""Tests for :class:`pyvisim.classic.Pipeline`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import ImageObj

#: Combined pipeline width: VLAD (``8 * 128``) concatenated with Fisher
#: (``2 * 8 * 128 + 8``).
PIPELINE_DIM = 8 * 128 + (2 * 8 * 128 + 8)

BATCH_SIZES = [2, 3, 4, 6, 16]


@pytest.fixture(scope="module")
def pipeline_embedders(
    category_train_images_flat: list[np.ndarray],
) -> tuple[VLADEmbedder, FisherVectorEmbedder]:
    """A learned VLAD and Fisher embedder, both sharing the default RootSIFT.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a ``(vlad, fisher)`` tuple of fitted embedders.
    """
    vlad = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    fisher = FisherVectorEmbedder(n_components=8, gmm_params={"rng": 0})
    vlad.learn(category_train_images_flat)
    fisher.learn(category_train_images_flat)
    return vlad, fisher


@pytest.fixture(scope="module")
def pipeline(
    pipeline_embedders: tuple[VLADEmbedder, FisherVectorEmbedder],
) -> Pipeline:
    """A pipeline combining the learned VLAD and Fisher embedders.

    :param pipeline_embedders: the ``(vlad, fisher)`` fixture.
    :returns: a ``Pipeline`` over both embedders.
    """
    vlad, fisher = pipeline_embedders
    return Pipeline([vlad, fisher])


def test_rejects_non_embedder() -> None:
    """A pipeline rejects members that are not embedders."""
    with pytest.raises(
        ValueError, match="only accepts instances of SerializableImageEmbedder"
    ):
        Pipeline(["not an embedder"])  # type: ignore[list-item]


def test_embed_concatenates(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """The pipeline concatenates each embedder's output along the feature axis."""
    out = pipeline.embed([checkerboard_image.array])
    assert out.shape == (1, PIPELINE_DIM)


def test_restores_flatten_flag(
    pipeline: Pipeline,
    pipeline_embedders: tuple[VLADEmbedder, FisherVectorEmbedder],
    checkerboard_image: ImageObj,
) -> None:
    """The pipeline restores each embedder's original ``flatten`` flag."""
    vlad, _ = pipeline_embedders
    original = vlad.flatten
    vlad.flatten = False
    try:
        pipeline.embed([checkerboard_image.array])
        assert vlad.flatten is False
    finally:
        vlad.flatten = original


def test_embed_batch(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """A batch of two images embeds to ``(2, PIPELINE_DIM)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert pipeline.embed(batch).shape == (2, PIPELINE_DIM)


def test_embed_accepts_tensor(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert pipeline.embed([tensor]).shape == (1, PIPELINE_DIM)


def test_similarity_score_shape(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """``similarity_score`` returns an ``(n1, n2)`` float32 matrix."""
    base = checkerboard_image.array
    images1 = [base, np.roll(base, 8, axis=0)]
    images2 = [np.roll(base, 8, axis=1)]
    scores = pipeline.similarity_score(images1, images2)
    assert scores.shape == (2, 1)
    assert scores.dtype == np.float32


def test_repr(pipeline: Pipeline) -> None:
    """``repr`` names the pipeline and each member embedder."""
    text = repr(pipeline)
    assert "Pipeline(" in text
    assert "VLADEmbedder(" in text
    assert "FisherVectorEmbedder(" in text


def test_to_dict_from_dict_round_trip(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """``Pipeline.from_dict(pipeline.to_dict())`` rebuilds an equivalent pipeline."""
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert isinstance(restored, Pipeline)
    assert len(restored.embedders) == len(pipeline.embedders)
    assert restored.similarity_func_name == pipeline.similarity_func_name
    probe = [checkerboard_image.array]
    assert np.allclose(restored.embed(probe), pipeline.embed(probe), atol=1e-5)


# Batching


@pytest.fixture(scope="module")
def sample_images(category_train_images_flat: list[np.ndarray]) -> list[np.ndarray]:
    """Four training images, enough for every batch size to split them.

    :param category_train_images_flat: flattened training images.
    :returns: a short list of images.
    """
    return category_train_images_flat[:4]


@pytest.fixture
def batched_pipeline(pipeline: Pipeline) -> Iterator[Pipeline]:
    """The pipeline, with its batch size restored after the test.

    :param pipeline: the pipeline shared by this module.
    :returns: the same pipeline, ready to be rebatched.
    """
    original = pipeline.batch_size
    yield pipeline
    pipeline.set_batch_size(original)


@pytest.fixture(scope="module")
def unbatched_results(
    pipeline: Pipeline, sample_images: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The embeddings and scores of the sample images, one image per batch.

    :param pipeline: the pipeline shared by this module.
    :param sample_images: the images to embed.
    :returns: the ``(embeddings, scores)`` every batch size has to reproduce.
    """
    original = pipeline.batch_size
    pipeline.set_batch_size(1)
    try:
        return (
            pipeline.embed(sample_images),
            pipeline.similarity_score(sample_images[:2], sample_images[2:]),
        )
    finally:
        pipeline.set_batch_size(original)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_the_embeddings(
    batched_pipeline: Pipeline,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the work: vectors and scores stay in place."""
    embeddings, scores = unbatched_results
    batched_pipeline.set_batch_size(batch_size)
    np.testing.assert_allclose(
        batched_pipeline.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_pipeline.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_whole_input_as_one_batch_does_not_change_the_embeddings(
    batched_pipeline: Pipeline,
    sample_images: list[np.ndarray],
    unbatched_results: tuple[np.ndarray, np.ndarray],
) -> None:
    """``-1`` embeds the input in one go, for the same vectors as ``1`` does."""
    embeddings, scores = unbatched_results
    batched_pipeline.set_batch_size(-1)
    np.testing.assert_allclose(
        batched_pipeline.embed(sample_images), embeddings, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        batched_pipeline.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_embedding_image_by_image_matches_the_whole_input(
    batched_pipeline: Pipeline, sample_images: list[np.ndarray]
) -> None:
    """One call per image stacks into what one call over the input returns."""
    batched_pipeline.set_batch_size(1)
    one_at_a_time = np.vstack(
        [batched_pipeline.embed([image]) for image in sample_images]
    )
    np.testing.assert_allclose(
        batched_pipeline.embed(sample_images), one_at_a_time, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(("n_images", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_images(
    batched_pipeline: Pipeline,
    category_train_images_flat: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    n_images: int,
    batch_size: int,
) -> None:
    """The last batch holds ``N % batch_size`` images, however small ``N`` is."""
    batch_lengths: list[int] = []
    embed_batch = batched_pipeline._embed

    def record(images: list[np.ndarray]) -> np.ndarray:
        batch_lengths.append(len(images))
        return embed_batch(images)

    monkeypatch.setattr(batched_pipeline, "_embed", record)
    batched_pipeline.set_batch_size(batch_size)
    batched_pipeline.embed(category_train_images_flat[:n_images])
    assert batch_lengths[-1] == n_images % batch_size
    assert sum(batch_lengths) == n_images


def test_embedding_matrix_shape_for_a_single_image(
    pipeline: Pipeline, sample_images: list[np.ndarray]
) -> None:
    """A single image keeps its own row of the embedding matrix."""
    assert pipeline.embed(sample_images[:1]).shape == (1, PIPELINE_DIM)


def test_embedding_nothing_raises(pipeline: Pipeline) -> None:
    """An input holding no image is an error, not an empty embedding matrix."""
    with pytest.raises(ValueError):
        pipeline.embed([])


def test_the_embedders_keep_their_own_batch_size(
    batched_pipeline: Pipeline, sample_images: list[np.ndarray]
) -> None:
    """A pipeline batch is split further by whatever its embedders are set to."""
    originals = [embedder.batch_size for embedder in batched_pipeline.embedders]
    for embedder in batched_pipeline.embedders:
        embedder.set_batch_size(2)
    batched_pipeline.set_batch_size(3)
    try:
        batched = batched_pipeline.embed(sample_images)
        batched_pipeline.set_batch_size(-1)
        whole = batched_pipeline.embed(sample_images)
    finally:
        for embedder, original in zip(
            batched_pipeline.embedders, originals, strict=True
        ):
            embedder.set_batch_size(original)
    np.testing.assert_allclose(batched, whole, rtol=1e-5, atol=1e-6)


def test_the_flatten_flag_is_restored_after_each_batch(
    batched_pipeline: Pipeline,
    pipeline_embedders: tuple[VLADEmbedder, FisherVectorEmbedder],
    sample_images: list[np.ndarray],
) -> None:
    """Forcing ``flatten`` on for the run must not leave the embedders changed."""
    vlad, _ = pipeline_embedders
    vlad.flatten = False
    batched_pipeline.set_batch_size(2)
    try:
        batched_pipeline.embed(sample_images)
        assert vlad.flatten is False
    finally:
        vlad.flatten = True


def test_batch_size_survives_a_file_round_trip(
    batched_pipeline: Pipeline, tmp_path: Path
) -> None:
    """A saved pipeline comes back with the batch size it was saved with."""
    batched_pipeline.set_batch_size(8)
    path = batched_pipeline.save_to_disk(tmp_path / "pipeline")
    assert Pipeline.load_from_disk(path).batch_size == 8
