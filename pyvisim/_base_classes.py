import abc
import logging
import pathlib
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import numpy as np

from ._utils import get_similarity_func
from .serialization import EMBEDDER_METADATA_KEY, SerializerMixin
from .typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    MatLike,
    SimilarityFunc,
)

#: Suffix of the files written by :meth:`SerializableImageEmbedder.save_to_disk`.
EMBEDDER_FILE_SUFFIX = ".embedder"


def _l2_normalize(vectors: FloatNumpyArray) -> FloatNumpyArray:
    """
    Scales every row of ``vectors`` to unit L2 length.

    A row of length zero carries no direction to preserve and is left as it is
    instead of being divided by zero.

    :param vectors: An ``(N, D)`` array of embeddings.
    :return: An ``(N, D)`` array whose non-zero rows have unit L2 norm.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return cast(
        FloatNumpyArray,
        np.divide(vectors, norms, out=vectors.copy(), where=norms > 0),
    )


class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity metrics.

    All concrete similarity metric classes must inherit from this class.

    Every metric processes its input in batches. What one batch holds depends
    on the metric (images, image pairs, ...).

    Setting ``batch_size=-1`` would treat the whole input as a single batch.

    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
        integer.
    """

    _logger = logging.getLogger("Similarity_Metrics")

    def __init__(self, batch_size: int = 16) -> None:
        self._batch_size: int
        # Assign via the property setter to trigger validation.
        self.batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = self._validate_batch_size(batch_size)

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets the number of items processed per batch.

        :param batch_size: Maximum number of images processed in a single
            batch. Set to ``-1`` to process all images as a single batch.
        :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
            integer.
        """
        self.batch_size = batch_size

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        """
        Raises ValueError if ``batch_size`` is neither ``-1`` nor a positive integer.
        """
        if not isinstance(batch_size, int):
            raise ValueError(
                f"batch_size must be an integer, got {type(batch_size).__name__}."
            )
        if batch_size != -1 and batch_size < 1:
            raise ValueError(
                "batch_size must be a positive integer or -1 (process the "
                f"whole input as one batch), got {batch_size}."
            )
        return batch_size

    @abc.abstractmethod
    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute the similarity scores matrix between two (batches of) images.

        :param images1: First (batch of) image(s) as ``MatLike`` (NumPy array,
            torch tensor or array-like).
        :param images2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: The similarity score matrix of shape ``(len(images1), len(images2))``.
        """
        pass


class FeatureExtractorBase(abc.ABC):
    """
    Abstract interface for extracting features from images.

    A feature extractor transforms an image (NumPy array) into a
    set of feature vectors (NumPy array).
    """

    _logger = logging.getLogger("Feature_Extractor")

    @abc.abstractmethod
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts features from an image.

        :param image: Input image as ``MatLike`` (NumPy array, torch tensor or
            array-like). It is normalized to a canonical ``uint8`` ``(H, W, C)``
            image before extraction.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: Feature descriptors (NumPy array).
        """
        pass

    def extract_batch(
        self,
        images: Sequence[MatLike],
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> list[Float32NumpyArray]:
        """
        Extracts features from a batch of images.

        Returns one ``(N_i, D)`` feature array per image, in input order, since
        the number of descriptors an image yields varies from image to image.
        This default implementation extracts one image at a time; extractors
        that can do the whole batch in one go (e.g. a single forward pass
        through a neural network) override it.

        :param images: Batch of images, each a ``MatLike`` (NumPy array, torch
            tensor or array-like) normalized to a canonical ``uint8``
            ``(H, W, C)`` image before extraction.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            It applies to every image of the batch. See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: One ``(N_i, D)`` feature array per input image.
        """
        return [self(image, dims=dims, value_range=value_range) for image in images]

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """
        The dimensionality (D) of each feature vector, i.e., shape[1] of the output.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise this feature extractor into a JSON-safe configuration dict.

        The dict captures the extractor's class name and the keyword arguments
        needed to rebuild an equivalent instance (see
        :func:`pyvisim.features.feature_extractor_from_dict`).

        :return: A mapping ``{"__class__": str, "config": dict}``.
        """
        return {
            "__class__": type(self).__name__,
            "config": self._serialization_config(),
        }

    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the JSON-safe constructor arguments needed to rebuild this extractor.

        Stateless extractors (e.g. SIFT, RootSIFT) need no configuration and
        return an empty mapping. Subclasses override this hook when they carry
        reconstructable parameters.

        :return: A JSON-safe mapping of constructor arguments.
        """
        return {}


class ImageEmbedderBase(SimilarityMetric):
    """
    Base class for all image embedders.

    An image embedder turns an image into a vector representation that can be
    used for indexing, retrieval, clustering or classification.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether :meth:`embed` L2-normalizes the embeddings it
        returns, so that they can be compared directly with a dot product.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``similarity_func`` is not a supported similarity
        metric, ``normalize`` is not a boolean, or ``batch_size`` is neither
        ``-1`` nor a positive integer.
    """

    def __init__(
        self,
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        # Set important attributes via setters to trigger error handling
        super().__init__(batch_size=batch_size)
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self._normalize: bool
        self.similarity_func = similarity_func
        self.normalize = normalize

    @property
    def normalize(self) -> bool:
        """Whether the embeddings returned by :meth:`embed` are L2-normalized."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        """Sets whether :meth:`embed` L2-normalizes the embeddings it returns."""
        if not isinstance(normalize, bool):
            raise ValueError(
                f"normalize must be a boolean, got {type(normalize).__name__}."
            )
        self._normalize = normalize

    @property
    def similarity_func(self) -> SimilarityFunc:
        """The resolved similarity function callable."""
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, name: str) -> None:
        """
        Resolves and stores a built-in similarity metric by name.

        :param name: One of ``"cosine"``, ``"euclidean"``, ``"l1"`` or
            ``"manhattan"``.
        :raises ValueError: If ``name`` is not a supported similarity metric.
        """
        self._similarity_func = get_similarity_func(name)
        self._similarity_func_name = name

    @property
    def similarity_func_name(self) -> str:
        """The name of the configured similarity metric (e.g. ``"cosine"``)."""
        return self._similarity_func_name

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embeds one or more images into a batch of vector representations.

        Each image is normalized to a canonical ``uint8`` ``(H, W, C)`` array
        before feature extraction, so NumPy arrays, torch tensors and other
        array-like inputs are all accepted. When a batch axis is present (via
        ``dims``), every image in the batch is embedded. The resulting vectors
        are L2-normalized row by row when :attr:`normalize` is True.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Consider using an iterator for large datasets.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: vector representations of the given images, L2-normalized
            row by row if :attr:`normalize` is True.
        """
        vectors = self._embed(images, dims=dims, value_range=value_range)
        return _l2_normalize(vectors) if self._normalize else vectors

    @abc.abstractmethod
    def _embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embeds one or more images, without the L2 normalization.

        Every subclass has to implement this method.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Consider using an iterator for large datasets.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: vector representations of the given images without L2 normalization.
        """
        raise NotImplementedError

    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        vector1 = self.embed(images1, dims=dims, value_range=value_range)
        vector2 = self.embed(images2, dims=dims, value_range=value_range)
        result = self.similarity_func(vector1, vector2)
        return np.asarray(result, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(similarity_func={self.similarity_func_name})"
        )


class SerializableImageEmbedder(ImageEmbedderBase, SerializerMixin):
    """
    Base for embedders that persist to a ``.embedder`` file.

    Adds the serialization contract of
    :class:`~pyvisim.serialization.SerializerMixin` on top of
    :class:`ImageEmbedderBase`: subclasses describe themselves as a JSON-safe
    state via :meth:`~pyvisim.serialization.SerializerMixin.to_dict` /
    :meth:`~pyvisim.serialization.SerializerMixin.from_dict`, and the mixin
    turns that state into a file and back. Both the classic embedders and the
    neural ones use this path, so a ``.embedder`` file is always a
    `safetensors <https://github.com/huggingface/safetensors>`_ file.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether :meth:`embed` L2-normalizes the embeddings it
        returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    _FILE_SUFFIX: ClassVar[str] = EMBEDDER_FILE_SUFFIX
    _METADATA_KEY: ClassVar[str] = EMBEDDER_METADATA_KEY
    _CLASS_KEY: ClassVar[str] = "embedder_class"

    #: Keys a serialised state must contain to be a valid embedder file.
    #: Subclasses extend this with their own required keys.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"embedder_class", "similarity_func", "normalize", "batch_size"}
    )

    def _restore_batch_size(self, state: dict[str, Any]) -> None:
        """
        Adopts the batch size recorded in a serialised state.

        :param state: A JSON-safe embedder description.
        :raises KeyError: If the state carries no batch size.
        :raises ValueError: If the stored batch size is neither ``-1`` nor a
            positive integer.
        """
        self.set_batch_size(state["batch_size"])

    def _restore_normalize(self, state: dict[str, Any]) -> None:
        """
        Adopts the normalization setting recorded in a serialised state.

        :param state: A JSON-safe embedder description.
        :raises KeyError: If the state carries no normalization setting.
        :raises ValueError: If the stored setting is not a boolean.
        """
        self.normalize = state["normalize"]

    @classmethod
    def _read_state(cls, path: pathlib.Path) -> dict[str, Any]:
        """
        Reads the embedder state a ``.embedder`` file holds.

        :param path: Path to the ``.embedder`` file.
        :return: The reconstructed embedder state, with arrays restored.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises ValueError: If the file is not a valid ``.embedder`` file.
        """
        try:
            return super()._read_state(path)
        except ValueError as error:
            raise ValueError(
                f"File {path} is not a valid {EMBEDDER_FILE_SUFFIX} file."
            ) from error
