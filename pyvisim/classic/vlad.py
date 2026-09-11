from typing import Any, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from ..typing import (
    Float32NumpyArray,
    IntNumpyArray,
)
from ._base_embedder import ClusteringBasedEmbedder
from ._clustering import PCA, ClusteringModelBase, KMeans


def _segment_sums(
    values: Float32NumpyArray, segment_ids: IntNumpyArray, n_segments: int
) -> Float32NumpyArray:
    """
    Sum the rows of ``values`` into the buckets named by ``segment_ids``.

    The rows are sorted by bucket once and summed with :func:`numpy.add.reduceat`,
    which keeps the whole reduction inside NumPy instead of adding row by row.
    Buckets no row falls into stay zero.

    :param values: An ``(N, D)`` array of rows to sum.
    :param segment_ids: An ``(N,)`` array of bucket indices in ``[0, n_segments)``.
    :param n_segments: Number of buckets.
    :return: An ``(n_segments, D)`` array holding the sum of each bucket.
    """
    sums = np.zeros((n_segments, values.shape[1]), dtype=values.dtype)
    if segment_ids.size == 0:
        return sums
    order = np.argsort(segment_ids, kind="stable")
    sorted_ids = segment_ids[order]
    # The first row of every run of equal ids starts a new bucket.
    starts = np.flatnonzero(np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1])))
    sums[sorted_ids[starts]] = np.add.reduceat(values[order], starts, axis=0)
    return sums


class VLADEmbedder(ClusteringBasedEmbedder):
    """
    This class embeds images into VLAD descriptor vectors
    using a chosen feature extractor and a K-Means clustering model,
    then compares two VLAD descriptor vectors with a user-specified
    or default (cosine) similarity function.

    The K-Means model is configured from the parameters passed to this
    constructor (``n_clusters`` plus the optional ``kmeans_params``
    dictionary) and fitted by calling :meth:`learn`. An optional PCA
    model for dimensionality reduction is configured the same way via
    ``pca_params``.

    The output when calling `embed` has shape (num_clusters * feature_dim,).

    You can use euclidean distance, manhattan distance, etc. as the similarity function.

    The embedding can be used for indexing, retrieval, clustering or classification tasks.

    :param feature_extractor: Feature extractor instance (should implement __call__).
        Defaults to RootSIFT.
    :param n_clusters: Number of K-Means clusters (visual words) to use.
    :param kmeans_params: Arguments for K-Means during vocabulary learning. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/classic/vlad/vlad.html#k-means-parameters-kmeans-params``.
    :param pca_params: Arguments for the Principal Component Analysis during vocabulary learning. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/classic/vlad/vlad.html#pca-parameters-pca-params``.
    :param power_norm_weight: Exponent for power normalization
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.
    :param normalize: Whether ``embed`` L2-normalizes the embeddings it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.

    .. rubric:: References

    - [1] Relja Arandjelović and Andrew Zisserman, 'All About VLAD', Department of Engineering Science, University of Oxford.
    - [2] Relja Arandjelović and Andrew Zisserman, "Three things everyone should know to improve object retrieval," Department of Engineering Science, University of Oxford.
    - [3] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """

    _clustering_model_cls = KMeans

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        n_clusters: int = 256,
        kmeans_params: dict[str, Any] | None = None,
        pca_params: dict[str, Any] | None = None,
        power_norm_weight: float = 1,  # no paper found where power norm weight is used for VLAD
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        similarity_func: str = "cosine",
        raise_error_when_pca_incompatible: bool = False,
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> None:
        if kmeans_params and "n_clusters" in kmeans_params:
            raise ValueError(
                "Pass 'n_clusters' directly to VLADEmbedder instead of inside kmeans_params."
            )
        clustering_model = KMeans(n_clusters=n_clusters, **(kmeans_params or {}))
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
            normalize=normalize,
            batch_size=batch_size,
        )

    @property
    def clustering_model(self) -> KMeans:
        return cast(KMeans, self._clustering_model)

    def _set_clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        if not isinstance(clustering_model, KMeans):
            raise ValueError(
                f"The clustering model must be an instance of pyvisim.classic._clustering.KMeans, not {type(clustering_model)}"
            )
        super()._set_clustering_model(clustering_model)

    def _encode_batch(
        self, descriptors: Float32NumpyArray, counts: IntNumpyArray
    ) -> Float32NumpyArray:
        """
        Encodes the stacked descriptors of one image batch into VLAD vectors.

        The whole batch is assigned to its nearest centroids in one call, and
        the residuals of every (image, cluster) pair are accumulated in one
        bucketed sum: an image's residual for a cluster is the sum of the
        descriptors assigned to it minus as many copies of the centroid.

        :param descriptors: The ``(N, D)`` descriptors of the batch, stacked in
            image order.
        :param counts: How many of the ``N`` rows belong to each image.
        :return: The ``(B, k * D)`` embeddings of the batch, or ``(B * k, D)``
            when ``flatten`` is off.
        :raises ValueError: If an image of the batch yields no descriptor.
        """
        if not counts.all():
            raise ValueError(
                "No descriptors found in the image. Cannot compute VLAD embedding."
            )
        descriptors = self._project(descriptors).astype(np.float32)
        centroids = self.clustering_model.cluster_centers
        n_images, k, dim = len(counts), len(centroids), descriptors.shape[1]

        labels = self.clustering_model.predict(descriptors)
        # One bucket per (image, cluster) pair, so a single sum covers the batch.
        image_ids = np.repeat(np.arange(n_images, dtype=np.intp), counts)
        buckets = image_ids * k + labels
        sums = _segment_sums(descriptors, buckets, n_images * k).reshape(
            n_images, k, dim
        )
        assigned = np.bincount(buckets, minlength=n_images * k).reshape(n_images, k)
        residuals = sums - assigned[..., np.newaxis].astype(np.float32) * centroids

        residuals = np.sign(residuals) * np.abs(residuals) ** self.power_norm_weight
        norms = (
            np.linalg.norm(residuals, axis=2, ord=self.norm_order, keepdims=True)
            + self.epsilon
        )
        residuals = residuals / norms
        shape = (n_images, k * dim) if self.flatten else (n_images * k, dim)
        return cast(Float32NumpyArray, residuals.reshape(shape))
