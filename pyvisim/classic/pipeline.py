import logging
from typing import Any, ClassVar, cast

import numpy as np

from .._base_classes import SerializableImageEmbedder
from ..typing import (
    FloatNumpyArray,
    UInt8NumpyArray,
)

#: On-disk format version of the serialised pipeline state.
_PIPELINE_FORMAT_VERSION = 3


class Pipeline(SerializableImageEmbedder):
    """
    A pipeline for computing feature vectors using a set of
    embedders.

    Currently, all vectors computed using the Embedders listed
    will always be flattened, because different Embedders also
    have different output sizes.

    :param embedders: A list of SerializableImageEmbedder instances.
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether :meth:`embed` L2-normalizes the joined embeddings
        it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    _logger = logging.getLogger("Pipeline")

    #: Keys a serialised state must contain to be a valid pipeline file.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"embedder_class", "classic", "similarity_func", "normalize", "batch_size"}
    )

    def __init__(
        self,
        embedders: list[SerializableImageEmbedder],
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        self._check_valid_embedders(embedders)
        self.embedders = embedders
        super().__init__(
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )

    def _check_valid_embedders(
        self, embedders: list[SerializableImageEmbedder]
    ) -> None:
        """
        Checks if all embedders in the pipeline are instances of SerializableImageEmbedder.
        :param embedders: list of embedders to check.
        """
        for embedder in embedders:
            if not isinstance(embedder, SerializableImageEmbedder):
                raise ValueError(
                    f"Pipeline only accepts instances of SerializableImageEmbedder, not {type(embedder)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": _PIPELINE_FORMAT_VERSION,
            "embedder_class": type(self).__name__,
            "classic": [embedder.to_dict() for embedder in self.embedders],
            "similarity_func": self._similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "Pipeline":
        cls._reject_unsupported_kwargs(kwargs)
        # Imported lazily to avoid an import cycle with the embedder registry.
        from ..serialization import embedder_from_dict

        embedders = [
            cast(SerializableImageEmbedder, embedder_from_dict(embedder_state))
            for embedder_state in state["classic"]
        ]
        pipeline = cls(embedders, similarity_func=state["similarity_func"])
        pipeline._restore_normalize(state)
        pipeline._restore_batch_size(state)
        return pipeline

    def _embed(self, images: list[UInt8NumpyArray]) -> FloatNumpyArray:
        all_embeddings = []
        for metric in self.embedders:
            # Each embedder has to be flattened to be usable here. Embedders that
            # do not expose a ``flatten`` flag are assumed to already emit flat
            # ``(num_imgs, feature_dim)`` embeddings.
            has_flatten = hasattr(metric, "flatten")
            if has_flatten:
                original_flatten = metric.flatten  # type: ignore[attr-defined]
                metric.flatten = True  # type: ignore[attr-defined]
            try:
                # Each of size (num_imgs, feature_dim)
                all_embeddings.append(metric.embed(images))
            finally:
                if has_flatten:
                    metric.flatten = original_flatten  # type: ignore[attr-defined]
        # The embedders' vectors sit side by side, in the pipeline's order.
        return np.hstack(all_embeddings)

    # def fit(self, images: Iterable[np.ndarray], reduce_dimension: bool = False, reduce_factor: int=2) -> None:
    #     """
    #     Trains any clustering model_files used by the embedders in this pipeline, if they have a fit method.
    #
    #     :param images: Iterable of images (NumPy arrays) used for fitting the pipeline's embedders.
    #     :param reduce_dimension: Whether to apply dimension reduction (e.g., PCA) if supported.
    #     :param reduce_factor: Factor to reduce the dimension by.
    #     """
    #     for metric in self.embedder:
    #         if hasattr(metric, 'fit') and callable(metric.fit):
    #             self._logger.info(f"Fitting {metric.__class__.__name__} with reduce_dimension={reduce_dimension}...")
    #             metric.fit(images, reduce_dimension=reduce_dimension, reduce_factor=reduce_factor)
    #         else:
    #             self._logger.warning(f"{metric.__class__.__name__} has no 'fit' method. Skipping...")

    def __repr__(self) -> str:
        """
        Returns a string representation of this Pipeline, including the names
        of the embedders and the similarity function used.
        """
        embedders_str = "\n".join([str(embedder) for embedder in self.embedders])
        return (
            f"Pipeline(\n"
            f"embedders=[{embedders_str}],\n"
            f"similarity_func={self._similarity_func_name})"
        )
