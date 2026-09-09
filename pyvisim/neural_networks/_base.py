"""
Shared base for the neural image embedders.
"""

import abc
from typing import Any, ClassVar, TypeVar, cast

from .._base_classes import SerializableImageEmbedder
from ..lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch

    from ..utils.torch_utils import decode_state_dict, encode_state_dict

_torch_import.check()

#: On-disk format version of the serialised neural embedder state.
_NEURAL_EMBEDDER_FORMAT_VERSION = 3

_NeuralEmbedderT = TypeVar("_NeuralEmbedderT", bound="NeuralImageEmbedder")


class NeuralImageEmbedder(SerializableImageEmbedder, torch.nn.Module):
    """
    Abstract base for image embedders backed by a torch module.

    It combines the two halves every neural embedder in pyvisim needs:
    :class:`~pyvisim._base_classes.SerializableImageEmbedder` contributes the
    ``similarity_func`` handling, the :meth:`similarity_score` built on top of
    :meth:`embed` and the ``.embedder`` file format, while
    :class:`torch.nn.Module` contributes the parameter, device and
    training-mode machinery.

    Subclasses implement :meth:`embed`, :meth:`_serialization_config` and
    :meth:`_from_config`, and must call this constructor before registering any
    submodule, so that ``torch.nn.Module`` is initialized first.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether ``embed`` L2-normalizes the embeddings it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    #: Keys a serialised state must contain to be a valid embedder file.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "embedder_class",
            "similarity_func",
            "normalize",
            "batch_size",
            "config",
            "state_dict",
        }
    )

    def __init__(
        self,
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> None:
        # torch.nn.Module.__init__ creates the attribute dicts that its
        # __setattr__ relies on, so it has to run before anything is assigned.
        torch.nn.Module.__init__(self)
        SerializableImageEmbedder.__init__(
            self,
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute, routing property assignments through the property.

        NOTE
        ----
        Since :class:`torch.nn.Module` overrides ``__setattr__`` and registers
        any ``torch.nn.Module`` value directly in ``self._modules``, assigning
        to a read-only property such as ``head`` would silently register an
        orphan submodule instead of failing. Routing assignments to class-level
        properties through ``property.__set__`` restores standard Python
        semantics. Hence, this override was necessary.

        :param name: Name of the attribute to set.
        :param value: Value to assign.
        :raises AttributeError: If ``name`` is a read-only property.
        """
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property):
            descriptor.__set__(self, value)
            return
        super().__setattr__(name, value)

    @abc.abstractmethod
    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the JSON-safe constructor arguments needed to rebuild this embedder.

        The configuration describes the *architecture* only (e.g. the backbone
        name or the CLIP variant); the learned weights travel separately as the
        model's ``state_dict``. It is the exact mapping :meth:`_from_config`
        consumes.

        :return: A JSON-safe mapping of constructor arguments.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _from_config(
        cls, config: dict[str, Any], **kwargs: Any
    ) -> "NeuralImageEmbedder":
        """
        Rebuild an embedder skeleton from a serialised configuration.

        Implementations must build the architecture *without* loading any
        pretrained weights: :meth:`from_dict` overwrites them right after with
        the serialised ``state_dict``.

        :param config: Mapping produced by :meth:`_serialization_config`.
        :param kwargs: Parameters the configuration alone cannot describe, forwarded
            from :meth:`~pyvisim._base_classes.SerializableImageEmbedder.load_from_disk`.
        :return: An embedder whose architecture matches ``config``.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": _NEURAL_EMBEDDER_FORMAT_VERSION,
            "embedder_class": type(self).__name__,
            "similarity_func": self._similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "config": self._serialization_config(),
            "state_dict": encode_state_dict(self),
        }

    @classmethod
    def from_dict(
        cls: type[_NeuralEmbedderT], state: dict[str, Any], **kwargs: Any
    ) -> _NeuralEmbedderT:
        embedder = cast(_NeuralEmbedderT, cls._from_config(state["config"], **kwargs))
        embedder.similarity_func = state["similarity_func"]
        embedder._restore_normalize(state)
        embedder._restore_batch_size(state)
        embedder.load_state_dict(decode_state_dict(state["state_dict"]))
        return embedder.eval()

    @property
    def device(self) -> "torch.device":
        """
        The device the model's parameters live on.

        Derived from the parameters themselves rather than cached, so it stays
        correct after the user moves the model with ``model.to(...)``.
        """
        return next(self.parameters()).device
