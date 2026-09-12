"""Tests for the embedder base class and its similarity-function helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim._errors import NotFittedError
from pyvisim.classic import FisherVectorEmbedder, VLADEmbedder
from pyvisim.classic._base_embedder import ClusteringBasedEmbedder
from pyvisim.classic._clustering import PCA
from pyvisim.features import Lambda, RootSIFT

if TYPE_CHECKING:
    from tests.conftest import ImageObj


# §3.1 similarity-function selection


def test_similarity_func_defaults_to_cosine() -> None:
    """Embedders default to the cosine similarity metric."""
    embedder = VLADEmbedder(n_clusters=8)
    assert embedder.similarity_func_name == "cosine"


@pytest.mark.parametrize("name", ["cosine", "euclidean", "l1", "manhattan"])
def test_similarity_func_accepts_supported_metrics(name: str) -> None:
    """Each supported metric name resolves to a callable."""
    embedder = VLADEmbedder(n_clusters=8, similarity_func=name)
    assert embedder.similarity_func_name == name
    assert callable(embedder.similarity_func)


def test_similarity_func_rejects_custom_function() -> None:
    """Passing a custom callable is no longer supported and raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported similarity function"):
        VLADEmbedder(n_clusters=8, similarity_func=lambda a, b: a)  # type: ignore[arg-type]


def test_similarity_func_rejects_unknown_name() -> None:
    """An unknown metric name raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported similarity function"):
        VLADEmbedder(n_clusters=8, similarity_func="dot")


# §3.1a the shared normalization flag


def test_normalize_rejects_non_boolean() -> None:
    """A non-boolean ``normalize`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="normalize must be a boolean"):
        VLADEmbedder(n_clusters=8, normalize=1)  # type: ignore[arg-type]


def test_normalize_off_keeps_the_raw_embedding(
    learned_vlad: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """With ``normalize`` off, ``embed`` returns the encoding unscaled."""
    raw = VLADEmbedder.from_dict({**learned_vlad.to_dict(), "normalize": False})
    norms = np.linalg.norm(raw.embed([checkerboard_image.array]), axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_normalize_on_scales_the_raw_embedding(
    learned_vlad: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """Normalizing only rescales the embedding, it does not turn it."""
    raw = VLADEmbedder.from_dict({**learned_vlad.to_dict(), "normalize": False})
    unscaled = raw.embed([checkerboard_image.array])
    scaled = learned_vlad.embed([checkerboard_image.array])
    assert np.linalg.norm(scaled, axis=1) == pytest.approx([1.0], rel=1e-5)
    assert scaled == pytest.approx(
        unscaled / np.linalg.norm(unscaled, axis=1, keepdims=True), rel=1e-5
    )


def test_normalize_survives_a_round_trip(
    learned_vlad: VLADEmbedder, tmp_path: Path
) -> None:
    """The normalization setting is part of the serialised embedder state."""
    raw = VLADEmbedder.from_dict({**learned_vlad.to_dict(), "normalize": False})
    reloaded = VLADEmbedder.load_from_disk(raw.save_to_disk(tmp_path / "model"))
    assert reloaded.normalize is False


# §3.1b loading a clustering model from a scikit-learn estimator


def test_load_clustering_model_from_sklearn_vlad() -> None:
    """A fitted ``sklearn.cluster.KMeans`` becomes VLAD's clustering model."""
    from sklearn.cluster import KMeans as SklearnKMeans

    rng = np.random.default_rng(0)
    features = rng.random((300, 128))  # RootSIFT output dimension
    sklearn_model = SklearnKMeans(n_clusters=8, random_state=0).fit(features)
    embedder = VLADEmbedder(n_clusters=8)
    embedder.load_clustering_model_from_sklearn(sklearn_model)
    assert embedder.clustering_model.is_fitted
    assert np.array_equal(
        embedder.clustering_model.cluster_centers, sklearn_model.cluster_centers_
    )


def test_load_clustering_model_from_sklearn_dim_mismatch_raises() -> None:
    """An estimator fitted on the wrong feature size is rejected."""
    from sklearn.cluster import KMeans as SklearnKMeans

    rng = np.random.default_rng(0)
    sklearn_model = SklearnKMeans(n_clusters=8, random_state=0).fit(
        rng.random((300, 16))
    )
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(RuntimeError, match="has to match"):
        embedder.load_clustering_model_from_sklearn(sklearn_model)


def test_load_clustering_model_from_sklearn_fisher() -> None:
    """A fitted diagonal ``GaussianMixture`` becomes Fisher's clustering model."""
    from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

    rng = np.random.default_rng(0)
    features = rng.random((300, 128))  # RootSIFT output dimension
    sklearn_model = SklearnGaussianMixture(
        n_components=8, covariance_type="diag", random_state=0
    ).fit(features)
    embedder = FisherVectorEmbedder(n_components=8)
    embedder.load_clustering_model_from_sklearn(sklearn_model)
    assert embedder.clustering_model.is_fitted
    assert np.array_equal(embedder.clustering_model.means, sklearn_model.means_)


# §3.2 ImageEmbedderBase shared behaviour (exercised via VLADEmbedder)


class _NoModelEmbedder(ClusteringBasedEmbedder):
    """Minimal concrete embedder used to test the "no clustering model" path."""

    def _encode_batch(self, descriptors: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Unused stub; ``learn`` fails before embedding is ever reached."""
        raise NotImplementedError


@pytest.fixture(scope="module")
def learned_vlad(category_train_images_flat: list[np.ndarray]) -> VLADEmbedder:
    """A ``VLADEmbedder(n_clusters=8)`` learned once for this module.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEmbedder``.
    """
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    embedder.learn(category_train_images_flat)
    return embedder


def test_default_feature_extractor_is_rootsift() -> None:
    """Embedders default to the RootSIFT feature extractor."""
    assert isinstance(VLADEmbedder(n_clusters=8).feature_extractor, RootSIFT)


def test_feature_extractor_setter_type_check() -> None:
    """Assigning a non-extractor to ``feature_extractor`` raises ``TypeError``."""
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(TypeError):
        embedder.feature_extractor = "x"  # type: ignore[assignment]


def test_pca_is_read_only() -> None:
    """``pca`` is a read-only property; direct assignment raises ``AttributeError``."""
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(AttributeError):
        embedder.pca = object()  # type: ignore[assignment]


def test_learn_without_clustering_model_raises(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Learning with no clustering model raises ``RuntimeError``."""
    embedder = _NoModelEmbedder(clustering_model=None)
    with pytest.raises(RuntimeError, match="no clustering model"):
        embedder.learn(category_train_images_flat)


def test_learn_bad_dim_reduction_factor_zero(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A zero ``dim_reduction_factor`` raises ``ValueError``."""
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    with pytest.raises(ValueError, match="must be a positive integer"):
        embedder.learn(category_train_images_flat, dim_reduction_factor=0)


def test_learn_bad_dim_reduction_factor_negative(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A negative ``dim_reduction_factor`` raises ``ValueError``."""
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    with pytest.raises(ValueError, match="must be a positive integer"):
        embedder.learn(category_train_images_flat, dim_reduction_factor=-2)


def test_learn_with_dim_reduction_factor_builds_pca(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A ``dim_reduction_factor`` builds and fits a PCA halving the dimension."""
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    embedder.learn(category_train_images_flat, dim_reduction_factor=2)
    assert isinstance(embedder.pca, PCA)
    assert embedder.pca.is_fitted is True
    assert embedder.pca.n_components == 64  # RootSIFT dim 128 // 2


def test_feature_extractor_pca_dim_mismatch_raises(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Assigning an extractor whose output dim mismatches the fitted PCA raises."""
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    embedder.learn(category_train_images_flat, dim_reduction_factor=2)  # PCA in=128
    with pytest.raises(RuntimeError):
        embedder.feature_extractor = Lambda(
            lambda image: np.ones((5, 64), np.float32), output_dim=64
        )


def test_save_unfitted_raises(tmp_path: Path) -> None:
    """Saving an unfitted embedder raises ``NotFittedError``."""
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(NotFittedError, match="not fitted"):
        embedder.save_to_disk(tmp_path / "m")


def test_save_appends_suffix(learned_vlad: VLADEmbedder, tmp_path: Path) -> None:
    """Saving appends the ``.embedder`` suffix and writes the file."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    assert str(path).endswith(".embedder")
    assert path.exists()


def test_save_keeps_existing_suffix(learned_vlad: VLADEmbedder, tmp_path: Path) -> None:
    """Saving with an existing ``.embedder`` suffix does not double it."""
    path = learned_vlad.save_to_disk(tmp_path / "m.embedder")
    assert path.name == "m.embedder"


def test_load_roundtrip_same_embedding(
    learned_vlad: VLADEmbedder, tmp_path: Path, checkerboard_image: ImageObj
) -> None:
    """A saved-and-reloaded embedder produces the same embedding."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    loaded = VLADEmbedder.load_from_disk(path)
    assert np.allclose(
        loaded.embed([checkerboard_image.array]),
        learned_vlad.embed([checkerboard_image.array]),
    )


def test_load_invalid_file_raises(tmp_path: Path) -> None:
    """Loading a file that is not a valid embedder raises ``ValueError``."""
    bad = tmp_path / "bad.embedder"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="not a valid .embedder file"):
        VLADEmbedder.load_from_disk(bad)


def test_load_rejects_deserialization_kwargs(
    learned_vlad: VLADEmbedder, tmp_path: Path
) -> None:
    """A classic embedder is fully described by its file and takes no extras."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    with pytest.raises(TypeError, match="'transform'"):
        VLADEmbedder.load_from_disk(path, transform=object())


def test_load_wrong_class_raises(learned_vlad: VLADEmbedder, tmp_path: Path) -> None:
    """Loading a VLAD file with the Fisher loader raises ``ValueError``."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    with pytest.raises(ValueError, match="was saved by VLADEmbedder"):
        FisherVectorEmbedder.load_from_disk(path)


def test_repr_smoke(learned_vlad: VLADEmbedder) -> None:
    """``repr`` names the embedder class and its RootSIFT feature extractor."""
    text = repr(learned_vlad)
    assert "VLADEmbedder(" in text
    assert "feature_extractor=RootSIFT" in text
