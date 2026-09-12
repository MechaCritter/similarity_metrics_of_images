"""Unit tests for :class:`pyvisim.neural_networks.ContrastiveSiameseNetwork`."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from pyvisim._errors import InvalidImageError
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import ContrastiveSiameseNetwork

from .._stubs import (
    FlattenBackbone,
    MeanBackbone,
    build_stub_model,
    install_head,
    make_identity_head,
    make_random_rgb_image,
    make_solid_rgb_image,
)

BATCH_SIZES = [2, 3, 4, 6, 16]

# §1 construction and backbone resolution


def test_unknown_backbone_name_raises() -> None:
    """An unrecognised backbone name raises ``ValueError`` at construction."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ContrastiveSiameseNetwork(backbone="alexnet")


def test_get_backbone_rejects_unknown_name() -> None:
    """The backbone resolver raises ``ValueError`` for unknown names."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ContrastiveSiameseNetwork._get_backbone(
            "definitely-not-a-backbone", pretrained=False
        )


def test_non_pretrained_backbone_builds_without_weights() -> None:
    """``pretrained_backbone=False`` builds a real ResNet-18 without weights."""
    model = ContrastiveSiameseNetwork(embedding_dim=8, pretrained_backbone=False)
    embeddings = model.embed(make_random_rgb_image(seed=1))
    assert embeddings.shape == (1, 8)


def test_default_embedding_dim_is_128() -> None:
    """The projection head defaults to a 128-dimensional embedding space."""
    model = build_stub_model(MeanBackbone())
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.out_features == 128


def test_head_input_matches_backbone_output_dim() -> None:
    """The head's ``in_features`` equals the backbone's ``output_dim``."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4)
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.in_features == MeanBackbone.output_dim


def test_device_attribute_is_torch_device() -> None:
    """The ``device`` string is normalized to a ``torch.device``."""
    model = build_stub_model(MeanBackbone(), device="cpu")
    assert model.device == torch.device("cpu")


# §2 forward: mathematical correctness on trivial examples


def test_forward_l2_normalizes_embeddings() -> None:
    """With identity backbone and head, ``forward`` is plain L2 normalization.

    The input row ``[3, 4]`` has norm 5, so the embedding must be
    ``[0.6, 0.8]``.
    """
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2)
    install_head(model, make_identity_head(2))
    embedding = model(torch.tensor([[3.0, 4.0], [0.0, 5.0]]))
    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    assert torch.allclose(embedding, expected, atol=1e-6)


def test_forward_output_has_unit_norm() -> None:
    """Every embedding row of a random batch has unit L2 norm."""
    torch.manual_seed(0)
    model = build_stub_model(FlattenBackbone(8), embedding_dim=5)
    embeddings = model(torch.randn(16, 8))
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones(16), atol=1e-5)


def test_forward_applies_head_before_normalizing() -> None:
    """With head weights ``diag(1, 2)``, input ``[1, 1]`` maps to ``[1, 2]/sqrt(5)``."""
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2)
    head = torch.nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        head.bias.zero_()
    install_head(model, head)
    embedding = model(torch.tensor([[1.0, 1.0]]))
    expected = torch.tensor([[1.0, 2.0]]) / math.sqrt(5.0)
    assert torch.allclose(embedding, expected, atol=1e-6)


# §3 embed: shapes, layouts and value ranges


@pytest.fixture
def contrastive_model() -> ContrastiveSiameseNetwork:
    """A small deterministic model for image-level tests.

    :return: Model with a :class:`MeanBackbone` and a 4-dim embedding space.
    """
    torch.manual_seed(0)
    return build_stub_model(MeanBackbone(), embedding_dim=4)


def test_embed_single_image_shape_and_norm(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """One HWC image embeds to a single unit-norm float32 row."""
    embeddings = contrastive_model.embed(make_random_rgb_image(seed=1))
    assert embeddings.shape == (1, 4)
    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_embed_list_of_images(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """A list of N images embeds to an ``(N, embedding_dim)`` batch."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2, 3)]
    embeddings = contrastive_model.embed(images)
    assert embeddings.shape == (3, 4)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0] * 3, abs=1e-5)


def test_embed_batch_matches_single_embedding(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """Batched embedding equals embedding each image individually."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2, 3)]
    batched = contrastive_model.embed(images)
    single = contrastive_model.embed(images[0])
    assert np.allclose(batched[0], single[0], atol=1e-6)


def test_embed_bhwc_array_matches_list(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """A stacked BHWC array yields the same embeddings as a list of images."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2)]
    from_list = contrastive_model.embed(images)
    from_batch = contrastive_model.embed(np.stack(images), dims="BHWC")
    assert np.array_equal(from_list, from_batch)


def test_embed_chw_matches_hwc(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """The same image in CHW layout yields the same embedding as in HWC."""
    image = make_random_rgb_image(seed=7)
    from_hwc = contrastive_model.embed(image)
    from_chw = contrastive_model.embed(image.transpose(2, 0, 1), dims="CHW")
    assert np.array_equal(from_hwc, from_chw)


def test_embed_unit_value_range_matches_uint8(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """A float image in [0, 1] embeds identically to its uint8 counterpart."""
    checker = np.zeros((8, 8, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    from_uint8 = contrastive_model.embed(checker)
    from_float = contrastive_model.embed(
        checker.astype(np.float64) / 255.0, value_range=(0.0, 1.0)
    )
    assert np.array_equal(from_uint8, from_float)


def test_embed_grayscale_image(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """A single-channel image is promoted to RGB and embedded normally."""
    gray = np.full((16, 16), 100, dtype=np.uint8)
    embeddings = contrastive_model.embed(gray, dims="HW")
    assert embeddings.shape == (1, 4)


def test_embed_empty_input_raises(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """An empty image collection raises ``ValueError``."""
    with pytest.raises(ValueError, match="at least one image"):
        contrastive_model.embed([])


def test_embed_rejects_string_input(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """A string is not an image and raises ``InvalidImageError``."""
    with pytest.raises(InvalidImageError):
        contrastive_model.embed("not-an-image.jpg")


def test_embed_is_deterministic(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """Embedding the same image twice yields identical embeddings."""
    image = make_random_rgb_image(seed=5)
    assert np.array_equal(
        contrastive_model.embed(image), contrastive_model.embed(image)
    )


def test_embed_restores_training_mode(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """``embed`` runs in eval mode but restores the previous training state."""
    contrastive_model.train()
    contrastive_model.embed(make_solid_rgb_image(value=10))
    assert contrastive_model.training is True
    contrastive_model.eval()
    contrastive_model.embed(make_solid_rgb_image(value=10))
    assert contrastive_model.training is False


# §4 embed: mathematical correctness through the full public path


def test_embed_of_uniform_image_is_uniform_unit_vector() -> None:
    """A constant image maps to the constant unit vector ``1/sqrt(D)``.

    With a plain ``ToTensor`` transform, a 2x2 uniform image becomes 12 equal
    feature values; the identity head then L2-normalizes them to
    ``1/sqrt(12)`` each.
    """
    model = build_stub_model(
        FlattenBackbone(12),
        embedding_dim=12,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(12))
    embeddings = model.embed(make_solid_rgb_image(value=51, size=2))
    assert embeddings.shape == (1, 12)
    assert np.allclose(embeddings, 1.0 / math.sqrt(12.0), atol=1e-6)


def test_embed_preserves_pixel_ratios() -> None:
    """Pixel intensities 30 and 40 embed as the 3:4 pattern of unit norm.

    A 1x2 image with pixel values 30 and 40 flattens channel-major to
    ``[30, 40, 30, 40, 30, 40]``; normalization cancels the ``/255`` scale,
    leaving exactly ``[3, 4, 3, 4, 3, 4] / (5 * sqrt(3))``.
    """
    model = build_stub_model(
        FlattenBackbone(6),
        embedding_dim=6,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(6))
    image = np.array([[[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    embeddings = model.embed(image)
    expected = np.array([3.0, 4.0] * 3) / (5.0 * math.sqrt(3.0))
    assert np.allclose(embeddings[0], expected, atol=1e-6)


# §5 similarity_score


def test_similarity_of_identical_images_is_one(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """Cosine similarity of an image with itself is 1."""
    image = make_random_rgb_image(seed=11)
    score = contrastive_model.similarity_score(image, image.copy())
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_similarity_matrix_shape(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """Two batches of sizes N and M produce an ``(N, M)`` similarity matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = contrastive_model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)


def test_similarity_scores_bounded_by_one(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """Cosine similarities of unit embeddings lie in ``[-1, 1]``."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = contrastive_model.similarity_score(batch_a, batch_b)
    assert np.all(score >= -1.0 - 1e-6)
    assert np.all(score <= 1.0 + 1e-6)


def test_similarity_equals_embedding_dot_product(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """On L2-normalized embeddings, cosine similarity is the dot product."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4)]
    score = contrastive_model.similarity_score(batch_a, batch_b)
    expected = contrastive_model.embed(batch_a) @ contrastive_model.embed(batch_b).T
    assert np.allclose(score, expected, atol=1e-5)


def test_similarity_is_symmetric(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """Swapping the inputs transposes the cosine similarity matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    forward_score = contrastive_model.similarity_score(batch_a, batch_b)
    backward_score = contrastive_model.similarity_score(batch_b, batch_a)
    assert np.allclose(forward_score, backward_score.T, atol=1e-6)


def test_named_similarity_func_is_resolved_and_used() -> None:
    """A named metric replaces the cosine default and is applied when scoring."""
    from pyvisim._utils import get_similarity_func

    model = build_stub_model(
        MeanBackbone(), embedding_dim=4, similarity_func="euclidean"
    )

    image1 = make_random_rgb_image(seed=1)
    image2 = make_random_rgb_image(seed=2)
    embeddings1 = model.embed(image1)
    embeddings2 = model.embed(image2)
    expected = np.asarray(get_similarity_func("euclidean")(embeddings1, embeddings2))
    assert np.allclose(model.similarity_score(image1, image2), expected)


def test_unknown_similarity_func_raises() -> None:
    """An unrecognised similarity metric name raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported similarity function"):
        build_stub_model(MeanBackbone(), embedding_dim=4, similarity_func="bogus")


# §6 head and backbone properties (read-only by design)


def test_head_property_returns_current_head(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """The property exposes the projection head module."""
    assert isinstance(contrastive_model.head, torch.nn.Linear)
    assert contrastive_model.head.out_features == 4


def test_backbone_property_returns_current_backbone(
    contrastive_model: ContrastiveSiameseNetwork,
) -> None:
    """The property exposes the feature-extraction backbone module."""
    assert isinstance(contrastive_model.backbone, MeanBackbone)


def test_head_is_read_only(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """Assigning to ``head`` raises instead of silently registering a module.

    ``torch.nn.Module.__setattr__`` would otherwise register the value as an
    orphan ``head`` submodule whose parameters get optimized but never used.
    """
    original_head = contrastive_model.head
    with pytest.raises(AttributeError):
        contrastive_model.head = torch.nn.Linear(3, 8)  # type: ignore[misc]
    assert "head" not in contrastive_model._modules
    assert contrastive_model.head is original_head


def test_backbone_is_read_only(contrastive_model: ContrastiveSiameseNetwork) -> None:
    """Assigning to ``backbone`` raises instead of silently registering a module."""
    original_backbone = contrastive_model.backbone
    with pytest.raises(AttributeError):
        contrastive_model.backbone = MeanBackbone()  # type: ignore[misc]
    assert "backbone" not in contrastive_model._modules
    assert contrastive_model.backbone is original_backbone


# §7 device handling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_embed_on_cuda_returns_cpu_numpy() -> None:
    """A CUDA model still returns NumPy embeddings on the host."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4, device="cuda")
    assert model.device.type == "cuda"
    embeddings = model.embed(make_random_rgb_image(seed=1))
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 4)


# §8 Oxford Flowers integration: real data through the stub backbone


def test_embed_on_flower_images_is_deterministic(
    contrastive_model: ContrastiveSiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Real images are no exception: embedding one twice gives the same rows."""
    image = flower_subset[0][0]
    embeddings = contrastive_model.embed(image)
    assert embeddings.shape == (1, 4)
    assert np.array_equal(embeddings, contrastive_model.embed(image))


def test_identical_flower_similarity_is_one(
    contrastive_model: ContrastiveSiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """A flower compared with itself has cosine similarity 1."""
    image = flower_subset[0][0]
    score = contrastive_model.similarity_score(image, image.copy())
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_similarity_matrix_on_flowers(
    contrastive_model: ContrastiveSiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Batches of flowers produce an ``(N, M)`` matrix of cosine similarities."""
    batch_a = [flower_subset[index][0] for index in (0, 1)]
    batch_b = [flower_subset[index][0] for index in (2, 3, 4)]
    score = contrastive_model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)
    assert np.all(score >= -1.0 - 1e-6)
    assert np.all(score <= 1.0 + 1e-6)


def test_flower_similarity_is_symmetric(
    contrastive_model: ContrastiveSiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Swapping two real flower images transposes the similarity matrix."""
    batch_a = [flower_subset[index][0] for index in (0, 1)]
    batch_b = [flower_subset[index][0] for index in (2, 3, 4)]
    forward_score = contrastive_model.similarity_score(batch_a, batch_b)
    backward_score = contrastive_model.similarity_score(batch_b, batch_a)
    assert np.allclose(forward_score, backward_score.T, atol=1e-6)


# §9 batching


@pytest.fixture
def sample_images() -> list[np.ndarray]:
    """Four distinct RGB images, enough for every batch size to split them.

    :return: Four ``(32, 32, 3)`` ``uint8`` arrays.
    """
    return [make_random_rgb_image(seed) for seed in (1, 2, 3, 4)]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_the_embeddings(
    contrastive_model: ContrastiveSiameseNetwork,
    sample_images: list[np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the forward passes: the rows stay in place."""
    contrastive_model.set_batch_size(1)
    one_by_one = contrastive_model.embed(sample_images)
    scores = contrastive_model.similarity_score(sample_images[:2], sample_images[2:])
    contrastive_model.set_batch_size(batch_size)
    np.testing.assert_allclose(
        contrastive_model.embed(sample_images), one_by_one, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        contrastive_model.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-6,
        atol=1e-6,
    )


def test_the_whole_input_as_one_batch_does_not_change_the_embeddings(
    contrastive_model: ContrastiveSiameseNetwork, sample_images: list[np.ndarray]
) -> None:
    """``-1`` embeds the input in one pass, for the same rows as ``1`` does."""
    contrastive_model.set_batch_size(1)
    one_by_one = contrastive_model.embed(sample_images)
    scores = contrastive_model.similarity_score(sample_images[:2], sample_images[2:])
    contrastive_model.set_batch_size(-1)
    np.testing.assert_allclose(
        contrastive_model.embed(sample_images), one_by_one, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        contrastive_model.similarity_score(sample_images[:2], sample_images[2:]),
        scores,
        rtol=1e-6,
        atol=1e-6,
    )


def test_embedding_image_by_image_matches_the_whole_input(
    contrastive_model: ContrastiveSiameseNetwork, sample_images: list[np.ndarray]
) -> None:
    """One call per image stacks into what one call over the input returns."""
    contrastive_model.set_batch_size(1)
    one_at_a_time = np.vstack(
        [contrastive_model.embed([image]) for image in sample_images]
    )
    np.testing.assert_allclose(
        contrastive_model.embed(sample_images), one_at_a_time, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(("n_images", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_images(
    contrastive_model: ContrastiveSiameseNetwork,
    monkeypatch: pytest.MonkeyPatch,
    n_images: int,
    batch_size: int,
) -> None:
    """The last batch holds ``N % batch_size`` images, however small ``N`` is."""
    batch_lengths: list[int] = []
    embed_batch = contrastive_model._embed

    def record(images: list[np.ndarray]) -> np.ndarray:
        batch_lengths.append(len(images))
        return embed_batch(images)

    monkeypatch.setattr(contrastive_model, "_embed", record)
    contrastive_model.set_batch_size(batch_size)
    contrastive_model.embed([make_random_rgb_image(seed) for seed in range(n_images)])
    assert batch_lengths[-1] == n_images % batch_size
    assert sum(batch_lengths) == n_images


def test_embedding_matrix_shape_for_a_single_image(
    contrastive_model: ContrastiveSiameseNetwork, sample_images: list[np.ndarray]
) -> None:
    """A single image keeps its own row of the embedding matrix."""
    assert contrastive_model.embed(sample_images[:1]).shape == (1, 4)


def test_batch_size_survives_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved network comes back with the batch size it was saved with."""
    network = ContrastiveSiameseNetwork(
        embedding_dim=4, pretrained_backbone=False, batch_size=3
    )
    path = network.save_to_disk(tmp_path / "network")
    assert ContrastiveSiameseNetwork.load_from_disk(path).batch_size == 3
