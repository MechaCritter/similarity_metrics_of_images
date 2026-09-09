from typing import Any, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from ..typing import (
    Float32NumpyArray,
    Float64NumpyArray,
    FloatNumpyArray,
    ImageInput,
    IntNumpyArray,
)
from ._base_embedder import ClusteringBasedEmbedder
from ._clustering import PCA, ClusteringModelBase, DiagCovarGaussianMixture


class FisherVectorEmbedder(ClusteringBasedEmbedder):
    """
    This class serves as an embedder that transforms input images into Fisher Vector descriptors.

    The Fisher Vector representation is based on the gradients of the GMM parameters
    (weights, means, and covariances) with respect to the feature descriptors extracted
    from the images. The representation is optionally power-normalized and L2-normalized.

    The Gaussian Mixture Model is configured from the parameters passed to
    this constructor (``n_components`` plus the optional ``gmm_params``
    dictionary) and fitted by calling :meth:`learn`. An optional PCA
    model for dimensionality reduction is configured the same way via
    ``pca_params``.

    The output when calling `embed` has shape (2 * num_clusters * feature_dim + num_clusters,).

    :param feature_extractor: Feature extractor instance. Default is RootSIFT
    :param n_components: Number of Gaussian mixture components (visual words) to use.
    :param gmm_params: Arguments for Gaussian Mixture Model during vocabulary learning. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/classic/fisher_vector/fisher_vector.html#gmm-parameters-gmm-params``.
    :param pca_params: Arguments for the Principal Component Analysis during vocabulary learning. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/classic/pca/pca.html#parameters-pca-params``.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed embedding vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.

    .. rubric:: References

    - [1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """

    _clustering_model_cls = DiagCovarGaussianMixture

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        n_components: int = 256,
        gmm_params: dict[str, Any] | None = None,
        pca_params: dict[str, Any] | None = None,
        power_norm_weight: float = 0.5,
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        similarity_func: str = "cosine",
        raise_error_when_pca_incompatible: bool = False,
        *,
        batch_size: int = 16,
    ):
        if gmm_params and "n_components" in gmm_params:
            raise ValueError(
                "Pass 'n_components' directly to FisherVectorEmbedder instead of inside gmm_params."
            )
        clustering_model = DiagCovarGaussianMixture(
            n_components=n_components, **(gmm_params or {})
        )
        pca = PCA(**pca_params) if pca_params is not None else None
        super().__init__(
            feature_extractor=feature_extractor,
            clustering_model=clustering_model,
            similarity_func=similarity_func,
            power_norm_weight=power_norm_weight,
            norm_order=norm_order,
            epsilon=epsilon,
            flatten=flatten,
            pca=pca,
            raise_error_when_pca_incompatible=raise_error_when_pca_incompatible,
            batch_size=batch_size,
        )

    @property
    def clustering_model(self) -> DiagCovarGaussianMixture:
        return cast(DiagCovarGaussianMixture, self._clustering_model)

    def _set_clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        if not isinstance(clustering_model, DiagCovarGaussianMixture):
            raise ValueError(
                f"The clustering model must be an instance of pyvisim.classic._clustering.DiagCovarGaussianMixture, not {type(clustering_model)}"
            )
        super()._set_clustering_model(clustering_model)

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float64NumpyArray:
        all_embeddings = [
            self._encode_batch(descriptors, counts)
            for descriptors, counts in self._iter_descriptor_batches(
                images, dims=dims, value_range=value_range
            )
        ]
        return np.vstack(all_embeddings)

    def _sufficient_statistics(
        self, descriptors: FloatNumpyArray, counts: IntNumpyArray
    ) -> tuple[Float64NumpyArray, Float64NumpyArray, Float64NumpyArray]:
        """
        Computes the mixture statistics the Fisher gradients are built from.

        The posterior probabilities of every descriptor in the batch come out
        of a single :meth:`predict_proba` call. Only the two weighted sums are
        taken one image at a time, since the images contribute different
        numbers of descriptors; each is a matrix product over all descriptors
        of that image.

        :param descriptors: The ``(N, D)`` descriptors of the batch, stacked in
            image order.
        :param counts: How many of the ``N`` rows belong to each image.
        :return: The per-image ``(B, k, 1)`` mean posterior and the two
            ``(B, k, D)`` posterior-weighted means of the descriptors and of
            their squares.
        """
        posterior_probabilities = self.clustering_model.predict_proba(descriptors)
        splits = np.cumsum(counts[:-1])
        statistics = [
            (
                probabilities.sum(axis=0),
                probabilities.T @ features,
                probabilities.T @ np.power(features, 2),
            )
            for probabilities, features in zip(
                np.split(posterior_probabilities, splits),
                np.split(descriptors, splits),
                strict=True,
            )
        ]
        per_image_counts = counts[:, np.newaxis, np.newaxis]
        pp_sum = np.stack([statistic[0] for statistic in statistics])[..., np.newaxis]
        return (
            pp_sum / per_image_counts,
            np.stack([statistic[1] for statistic in statistics]) / per_image_counts,
            np.stack([statistic[2] for statistic in statistics]) / per_image_counts,
        )

    def _encode_batch(
        self, descriptors: Float32NumpyArray, counts: IntNumpyArray
    ) -> Float64NumpyArray:
        """
        Encodes the stacked descriptors of one image batch into Fisher vectors.

        The gradients with respect to the mixture weights, means and
        covariances, the power normalization and the L2 normalization all run
        over the whole batch at once, on the statistics
        :meth:`_sufficient_statistics` collects.

        :param descriptors: The ``(N, D)`` descriptors of the batch, stacked in
            image order.
        :param counts: How many of the ``N`` rows belong to each image.
        :return: The ``(B, 2 * k * D + k)`` Fisher vectors of the batch.
        """
        descriptors = self._project(descriptors)
        mixture_weights = self.clustering_model.weights
        means = self.clustering_model.means
        covariances = self.clustering_model.covariances

        pp_sum, pp_x, pp_x_2 = self._sufficient_statistics(descriptors, counts)

        # Compute GMM gradients wrt its parameters
        d_pi = pp_sum.squeeze(axis=-1) - mixture_weights

        d_mu = pp_x - pp_sum * means

        d_sigma_t1 = pp_sum * np.power(means, 2)
        d_sigma_t2 = pp_sum * covariances
        d_sigma_t3 = 2 * pp_x * means
        d_sigma = -pp_x_2 - d_sigma_t1 + d_sigma_t2 + d_sigma_t3

        # Apply analytical diagonal normalization
        sqrt_mixture_weights = np.sqrt(mixture_weights)
        d_pi /= sqrt_mixture_weights
        d_mu /= sqrt_mixture_weights[:, np.newaxis] * np.sqrt(covariances)
        d_sigma /= np.sqrt(2) * sqrt_mixture_weights[:, np.newaxis] * covariances

        # Concatenate GMM gradients to form Fisher vector representation
        n_images = len(counts)
        descriptor_vectors = np.hstack(
            (
                d_pi,
                d_mu.reshape(n_images, -1),
                d_sigma.reshape(n_images, -1),
            )
        )

        # Power normalization and L2 normalization
        descriptor_vectors = np.sign(descriptor_vectors) * np.power(
            np.abs(descriptor_vectors), self.power_norm_weight
        )
        norms = (
            np.linalg.norm(
                descriptor_vectors, axis=1, ord=self.norm_order, keepdims=True
            )
            + self.epsilon
        )
        return cast(Float64NumpyArray, descriptor_vectors / norms)
