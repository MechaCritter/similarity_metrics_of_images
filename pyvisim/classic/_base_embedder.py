import warnings
from collections.abc import Iterator
from typing import Any, ClassVar, TypeVar

import numpy as np

from .._base_classes import FeatureExtractorBase, SerializableImageEmbedder
from .._config import setup_logging
from .._errors import NotFittedError
from ..features._registry import feature_extractor_from_dict
from ..features._root_sift import RootSIFT
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    IntNumpyArray,
    UInt8NumpyArray,
)
from ..utils.image_utils import iter_image_batches
from ._clustering import PCA, ClusteringModelBase

setup_logging()

_CLUSTERING_EMBEDDER_FILE_FORMAT_VERSION = 3
_CLUSTERING_EMBEDDER_FILE_FORMAT_VERSION_COMPATIBILITY: dict[tuple[int, int], bool] = {
    # Version 2 adds the "batch_size" key, which version 1 ignores but version 2
    # requires, so the compatibility only holds in one direction.
    (1, 2): True,  # version 1 can read files from version 2
    (2, 1): False,  # version 2 cannot read files from version 1
    # Version 3 adds the "normalize" key the same way.
    (1, 3): True,  # version 1 can read files from version 3
    (2, 3): True,  # version 2 can read files from version 3
    (3, 1): False,  # version 3 cannot read files from version 1
    (3, 2): False,  # version 3 cannot read files from version 2
    #
    # TODO: when the next _CLUSTERING_EMBEDDER_FILE_FORMAT_VERSION comes, check if
    # it's forward / backward compatible, then add entries like the ones above.
}
_ClusteringEmbedderT = TypeVar("_ClusteringEmbedderT", bound="ClusteringBasedEmbedder")


class FeatureBasedEmbedder(SerializableImageEmbedder):
    """
    Base class for embedders that derive their vector representation from local
    features extracted by a :class:`~pyvisim.features.FeatureExtractorBase`
    (e.g. SIFT, SURF or deep features).

    :param feature_extractor: Feature extractor instance (should implement
        ``__call__``). Defaults to :class:`~pyvisim.features.RootSIFT`.
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether ``embed`` L2-normalizes the embeddings it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        self._feature_extractor: FeatureExtractorBase
        super().__init__(
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else RootSIFT()
        )

    @property
    def feature_extractor(self) -> FeatureExtractorBase:
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, feature_extractor: FeatureExtractorBase) -> None:
        self._validate_feature_extractor(feature_extractor)
        self._feature_extractor = feature_extractor

    def _validate_feature_extractor(
        self, feature_extractor: FeatureExtractorBase
    ) -> None:
        """
        Validates a candidate feature extractor before it is stored.

        :param feature_extractor: Feature extractor to validate.
        :raises TypeError: If ``feature_extractor`` is not a
            :class:`~pyvisim.features.FeatureExtractorBase` instance.
        """
        if not isinstance(feature_extractor, FeatureExtractorBase):
            raise TypeError(
                f"feature_extractor must be an instance of FeatureExtractorBase, not {type(feature_extractor)}"
            )


class ClusteringBasedEmbedder(FeatureBasedEmbedder):
    """
    Base class for image embedders that aggregate local features with a
    clustering model. Subclasses (VLAD and Fisher Vector) use a combination of:

    - A feature extractor: Extract local features from an image (e.g. SIFT, SURF, or Deep Features).
    - A clustering model (K-Means for VLAD or GMM for Fisher Vector): aggregates local features to their
    nearest centroids to produce fix-sized vectors.
    - A similarity function: computes a single float value from the vector representations that represents
    the similarity between two images.

    The embedding can be used for indexing, retrieval, clustering or classification tasks.
    :param feature_extractor: Feature extractor instance (should implement __call__).
        Defaults to RootSIFT.
    :param clustering_model: Clustering model used for generating descriptors.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
    ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param pca: PCA model for dimensionality reduction (optional). Subclasses build
    it from the ``pca_params`` dictionary passed to their constructors.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, an Error will be raised
    :param normalize: Whether ``embed`` L2-normalizes the embeddings it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    _clustering_model_cls: ClassVar[type[ClusteringModelBase]]

    #: Keys that a serialised state must contain to be a valid embedder file.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "embedder_class",
            "clustering_model",
            "pca",
            "power_norm_weight",
            "norm_order",
            "epsilon",
            "flatten",
            "raise_error_when_pca_incompatible",
            "similarity_func",
            "normalize",
            "batch_size",
            "feature_extractor",
        }
    )

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        clustering_model: ClusteringModelBase | None = None,
        similarity_func: str = "cosine",
        power_norm_weight: float = 1,
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        pca: PCA | None = None,
        raise_error_when_pca_incompatible: bool = True,
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        # Set important attributes via setters to trigger error handling
        self._clustering_model: ClusteringModelBase | None = None
        self._pca: PCA | None = None

        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.raise_error_when_pca_incompatible = raise_error_when_pca_incompatible

        # The feature extractor setter validates against the (currently unset)
        # PCA / clustering model, so both must already exist as ``None`` above.
        super().__init__(
            feature_extractor=feature_extractor,
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )

        if pca is not None:
            self._set_pca(pca)
        if clustering_model is not None:
            self._set_clustering_model(clustering_model)

    def _validate_feature_extractor(
        self, feature_extractor: FeatureExtractorBase
    ) -> None:
        """
        Validates a candidate feature extractor against the configured PCA or
        clustering model before it is stored.

        In addition to the type check performed by the base class, the feature
        extractor output size must match the input size of the fitted PCA (if
        any) or otherwise the fitted clustering model.

        :param feature_extractor: Feature extractor to validate.
        :raises TypeError: If ``feature_extractor`` is not a
            :class:`~pyvisim.features.FeatureExtractorBase` instance.
        :raises RuntimeError: If its output size is incompatible with the
            fitted PCA or clustering model.
        """
        super()._validate_feature_extractor(feature_extractor)
        if self._pca is not None and self._pca.is_fitted:
            if feature_extractor.output_dim != self._pca.n_features_in:
                raise RuntimeError(
                    f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                    f"But PCA accepts input dim {self._pca.n_features_in}"
                )
        else:
            if self._clustering_model is not None and self._clustering_model.is_fitted:
                if feature_extractor.output_dim != self._clustering_model.n_features_in:
                    raise RuntimeError(
                        f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                        f"But clustering model accepts input dim {self._clustering_model.n_features_in}"
                    )

    @property
    def clustering_model(self) -> ClusteringModelBase | None:
        return self._clustering_model

    def _set_clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        """
        Validates the given clustering model against the current PCA or
        feature extractor and stores it.

        Dimension checks only apply to fitted models; an unfitted model is
        stored as-is and validated once it is fitted via :meth:`learn`.

        :param clustering_model: Clustering model to validate and store.
        """
        if not clustering_model.is_fitted:
            self._clustering_model = clustering_model
            return
        if self._pca is not None and self._pca.is_fitted:
            if self._pca.n_components != clustering_model.n_features_in:
                if self.raise_error_when_pca_incompatible:
                    raise RuntimeError(
                        f"PCA is incompatible with the new clustering model. "
                        f"PCA input size: {self._pca.n_components}, "
                        f"New clustering model input size: {clustering_model.n_features_in}. "
                        f"If you want the PCA to be reset to None instead, set raise_error_when_pca_incompatible=False."
                    )
                warnings.warn(
                    f"PCA is incompatible with the new clustering model. "
                    f"PCA input size: {self._pca.n_components}, "
                    f"New clustering model input size: {clustering_model.n_features_in}. "
                    "PCA will be reset to None to avoid errors."
                    "If you want to raise an Error instead when this happens, set raise_error_when_pca_incompatible=False.",
                    stacklevel=2,
                )
                self._pca = None
        else:
            if self._feature_extractor.output_dim != clustering_model.n_features_in:
                raise RuntimeError(
                    "Feature extractor output size has to match the clustering model input size. "
                    f"Feature extractor has output size {self._feature_extractor.output_dim}, "
                    f"while clustering model has input size {clustering_model.n_features_in}"
                )
        self._clustering_model = clustering_model

    @property
    def pca(self) -> PCA | None:
        return self._pca

    def _set_pca(self, pca: PCA) -> None:
        """
        Validates and stores the given PCA model.

        :param pca: PCA model to validate and store.
        :raises ValueError: If ``pca`` is not a ``PCA`` instance or its dimensions are
            incompatible with the feature extractor or clustering model.
        """
        if not isinstance(pca, PCA):
            raise ValueError(
                f"The PCA model must be an instance of pyvisim.classic._clustering.PCA, not {type(pca)}"
            )
        if not pca.is_fitted:
            self._pca = pca
            return
        if pca.n_features_in != self._feature_extractor.output_dim:
            raise ValueError(
                "PCA input size has to match the feature extractor output size. "
                f"PCA model has input size {pca.n_features_in}, "
                f"while feature extractor has output size {self._feature_extractor.output_dim}"
            )

        if self._clustering_model is not None and self._clustering_model.is_fitted:
            if pca.n_components != self._clustering_model.n_features_in:
                raise ValueError(
                    "PCA input size has to match the clustering model input size."
                    f"PCA model has input size {pca.n_components}, "
                    f"while clustering model has input size {self._clustering_model.n_features_in}"
                )

        self._pca = pca

    def _iter_descriptor_batches(
        self,
        images: ImageInput,
        *,
        dims: str,
        value_range: tuple[float, float],
    ) -> Iterator[tuple[Float32NumpyArray, IntNumpyArray]]:
        """
        Generator that yields the stacked descriptors of one image batch at a
        time.

        """
        for batch in iter_image_batches(
            images, self.batch_size, dims=dims, value_range=value_range
        ):
            yield self._extract_descriptors(batch)

    def _extract_descriptors(
        self, images: list[UInt8NumpyArray]
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Extracts the local descriptors of one image batch and stacks them.

        Every image of the batch reaches the feature extractor in a single
        :meth:`~pyvisim._base_classes.FeatureExtractorBase.extract_batch` call,
        and the descriptors it returns are concatenated into one ``(N, D)``
        array. Whatever runs next (the PCA, the clustering model) therefore
        sees the batch as one matrix instead of one image at a time. The
        descriptors are the raw extractor output: the PCA is left to
        :meth:`_project`, since :meth:`learn` has to fit it on them first.

        :param images: One batch of canonical ``uint8`` images of shape
            ``(H, W[, C])``.
        :return: The ``(N, D)`` descriptors of the batch, stacked in image
            order, and how many of the ``N`` rows belong to each image.
        """
        # The batch holds canonical uint8 (H, W[, C]) images already, so the
        # extractor is called with the default dims and value range.
        per_image = self.feature_extractor.extract_batch(images)
        counts = np.array([len(features) for features in per_image], dtype=np.intp)
        return np.concatenate(per_image), counts

    def _project(self, descriptors: FloatNumpyArray) -> FloatNumpyArray:
        """Reduces descriptors with the configured PCA, if there is one."""
        if self.pca:
            return self.pca.transform(descriptors.astype(np.float32))
        return descriptors

    def learn(
        self,
        images: ImageInput,
        /,
        *,
        dim_reduction_factor: int | None = None,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        """
        Learns the visual vocabulary from the given images.

        The clustering model configured at initialization is fitted
        on the extracted features. If a PCA model is configured, the features
        are reduced with it first (fitting it beforehand if necessary).

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Each image is normalized to a canonical
            ``uint8`` ``(H, W, C)`` array before feature extraction.
        :param dim_reduction_factor: If a value is provided, a new PCA model will be used to reduce the dimensionality of the feature space
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :raises RuntimeError: If the embedder has no clustering model configured.
        :raises ValueError: If dim_reduction_factor is provided but is not a positive integer.
        """
        if dim_reduction_factor is not None and (
            dim_reduction_factor <= 0 or not isinstance(dim_reduction_factor, int)
        ):
            raise ValueError("dim_reduction_factor must be a positive integer.")
        if self._clustering_model is None:
            raise RuntimeError(
                "This embedder has no clustering model to fit. "
                "Configure one via the constructor parameters."
            )
        features: FloatNumpyArray = np.vstack(
            [
                descriptors
                for descriptors, _ in self._iter_descriptor_batches(
                    images, dims=dims, value_range=value_range
                )
            ]
        )
        print("[INFO] Learning the visual vocabulary with the following parameters:")
        print("   - Number of clusters:", self._clustering_model.n_clusters)
        print("   - Feature Extractor used:", self.feature_extractor.__class__.__name__)
        print("   - Dimension of the feature space:", feat_dim := features.shape[1])
        if dim_reduction_factor:
            if (n_components := feat_dim // dim_reduction_factor) <= 0:
                raise ValueError(
                    f"dim_reduction_factor {dim_reduction_factor} is too large for the feature dimension {feat_dim}. "
                    f"Resulting PCA components would be {n_components}. Please choose a smaller dim_reduction_factor."
                )
            self._pca = PCA(n_components=n_components)
        if self._pca is not None:
            if not self._pca.is_fitted:
                self._pca.fit(features)
            features = self._pca.transform(features)
            print("   - New dimension after PCA reduction:", self._pca.n_components)
        self._clustering_model.fit(features)

    def load_clustering_model_from_sklearn(self, model: Any) -> None:
        """
        Replaces this embedder's clustering model with one created from a
        scikit-learn estimator.

        The estimator's constructor arguments are translated by the
        clustering model class' ``from_sklearn`` method; if the estimator is
        fitted, its learned state is adopted and validated against the
        configured feature extractor / PCA. Only clustering model classes
        exposing ``from_sklearn`` support this — currently
        :class:`~pyvisim.classic._clustering.KMeans` (VLAD) and
        :class:`~pyvisim.classic._clustering.DiagCovarGaussianMixture`
        (Fisher Vector).

        :param model: A scikit-learn estimator of the type matching this
            embedder's clustering model (e.g. :class:`sklearn.cluster.KMeans`
            for VLAD), fitted or not.
        :raises NotImplementedError: If this embedder's clustering model class
            cannot be created from a scikit-learn estimator.
        :raises TypeError: If ``model`` is not of the supported estimator type.
        :raises RuntimeError: If the fitted estimator's input size is
            incompatible with the configured feature extractor or PCA.
        """
        from_sklearn = getattr(self._clustering_model_cls, "from_sklearn", None)
        if from_sklearn is None:
            raise NotImplementedError(
                f"{self._clustering_model_cls.__name__} cannot be created from "
                "a scikit-learn estimator."
            )
        self._set_clustering_model(from_sklearn(model))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the learned embedder into a JSON-safe state dictionary.

        The returned mapping describes the embedder class, its clustering model,
        optional PCA, normalization hyperparameters, similarity metric and
        feature extractor. Arrays are kept as ``__ndarray__`` nodes so the
        whole dictionary can be embedded inside a larger state (e.g. an
        :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`).

        :return: A JSON-safe embedder description suitable for
            :meth:`from_dict`.
        :raises NotFittedError: If the clustering model is missing or not fitted.
        """
        if self._clustering_model is None or not self._clustering_model.is_fitted:
            raise NotFittedError(
                "Cannot serialise an embedder whose clustering model is not "
                "fitted. Call 'learn' first."
            )
        return {
            "format_version": _CLUSTERING_EMBEDDER_FILE_FORMAT_VERSION,
            "embedder_class": type(self).__name__,
            "clustering_model": self._clustering_model.to_dict(),
            "pca": self._pca.to_dict() if self._pca is not None else None,
            "power_norm_weight": self.power_norm_weight,
            "norm_order": self.norm_order,
            "epsilon": self.epsilon,
            "flatten": self.flatten,
            "raise_error_when_pca_incompatible": self.raise_error_when_pca_incompatible,
            "similarity_func": self._similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "feature_extractor": self._feature_extractor.to_dict(),
        }

    @classmethod
    def from_dict(
        cls: type[_ClusteringEmbedderT], state: dict[str, Any], **kwargs: Any
    ) -> _ClusteringEmbedderT:
        cls._reject_unsupported_kwargs(kwargs)
        embedder = cls(
            feature_extractor=feature_extractor_from_dict(state["feature_extractor"]),
            similarity_func=state["similarity_func"],
            power_norm_weight=state["power_norm_weight"],
            norm_order=state["norm_order"],
            epsilon=state["epsilon"],
            flatten=state["flatten"],
            raise_error_when_pca_incompatible=state[
                "raise_error_when_pca_incompatible"
            ],
        )
        embedder._restore_normalize(state)
        embedder._restore_batch_size(state)
        if state["pca"] is not None:
            embedder._set_pca(PCA.from_dict(state["pca"]))
        embedder._set_clustering_model(
            cls._clustering_model_cls.from_dict(state["clustering_model"])
        )
        return embedder

    def __repr__(self) -> str:
        n_clusters = (
            self._clustering_model.n_clusters if self._clustering_model else None
        )
        return (
            self.__class__.__name__
            + f"(feature_extractor={self.feature_extractor.__class__.__name__}, \n"
            f"similarity_func={self.similarity_func_name}, \n"
            f"Number of cluster={n_clusters}, \n"
            f"Power Norm Weight={self.power_norm_weight}, \n"
            f"Norm Order={self.norm_order})"
        )
