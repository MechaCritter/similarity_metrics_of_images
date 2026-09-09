"""CLIP image embedder built on pyvisim's own CLIP implementation.

The embedder pairs the image towers implemented in :mod:`._model` with
pretrained safetensors weights downloaded from the Hugging Face Hub (see
:mod:`._registry`), so no third-party CLIP library is needed. Every variant
of open_clip's registry whose image tower is a standard CLIP Vision
Transformer or modified ResNet, including all original OpenAI models, is
supported.
"""

from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from PIL import Image

from ..._base_classes import SerializableImageEmbedder
from ...lazy_import import OptionalImport
from ...typing import (
    Float32NumpyArray,
    ImageInput,
    UInt8NumpyArray,
)
from ...utils.image_utils import iter_image_batches
from ._registry import (
    CheckpointSpec,
    VisionConfig,
    fetch_checkpoint,
    get_checkpoint_spec,
    get_model_config,
)

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

    from ...utils.torch_utils import (
        decode_state_dict,
        encode_state_dict,
        resolve_device,
    )
    from ._model import build_vision_model, load_vision_weights

_torch_import.check()

#: On-disk format version of the serialised CLIP embedder state.
_CLIP_EMBEDDER_FORMAT_VERSION = 3


def _build_preprocess(
    config: VisionConfig, spec: CheckpointSpec
) -> "transforms.Compose":
    """
    Build the CLIP inference preprocessing pipeline of a checkpoint.

    Matches the transform open_clip builds for the checkpoint: a bicubic
    resize (of the shortest side plus a center crop, or of both sides for
    ``"squash"`` checkpoints), conversion to a ``[0, 1]`` tensor and
    normalization with the checkpoint's channel statistics.

    :param config: Architecture description carrying the input size.
    :param spec: Checkpoint metadata carrying the normalization statistics
        and resize mode.
    :return: The composed torchvision transform.
    """
    size = config.image_size
    if spec.resize_mode == "squash":
        resize: list[transforms.Compose | object] = [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            )
        ]
    else:
        resize = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
        ]
    return transforms.Compose(
        [*resize, transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    )


class ClipEmbedder(SerializableImageEmbedder):
    """
    Embeds images with a pretrained CLIP model.

    The safetensors checkpoint of the requested ``variant`` and
    ``pretrained`` tag is downloaded from the Hugging Face Hub on first use
    and cached (see :func:`pyvisim.neural_networks.clip.fetch_checkpoint`);
    only the image tower is loaded, and it always runs in ``float32``.
    :meth:`embed` returns one embedding per image; embeddings are
    L2-normalized by default so they can be compared directly with a dot
    product or the cosine similarity metric.

    Variant names and pretrained tags follow open_clip, e.g.
    ``ClipEmbedder("ViT-B-32", pretrained="laion2b_s34b_b79k")``. OpenAI-style
    variant spellings (``"ViT-B/32"``) are accepted as aliases. See
    :func:`pyvisim.neural_networks.clip.available_variants` and
    :func:`pyvisim.neural_networks.clip.available_pretrained` for the
    supported combinations.

    :param variant: CLIP variant name (default ``"ViT-B-32"``).
    :param pretrained: Pretrained tag naming the weights (default
        ``"openai"``, the original OpenAI checkpoint of the variant).
    :param device: Device to run the model on (``"cpu"`` or ``"cuda"``).
        Defaults to ``"cuda"`` when a CUDA device is available, else
        ``"cpu"``. The model runs in ``float32`` on either device.
    :param normalize: Whether to L2-normalize the returned embeddings.
    :param similarity_func: Name of the built-in similarity metric used to
        score two embeddings. One of ``"cosine"`` (default), ``"euclidean"``,
        ``"l1"`` or ``"manhattan"``.
    :param cache_dir: Directory of the Hugging Face Hub cache the checkpoint
        is stored in. Defaults to the standard Hub cache
        (``~/.cache/huggingface/hub``), so weights already downloaded via
        open_clip's Hub downloads are reused.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``variant``, ``pretrained`` or
        ``similarity_func`` is not supported, or the checkpoint does not
        match the architecture.
    :raises ImportError: If the ``nn`` extra is not installed.
    :raises huggingface_hub.errors.HfHubHTTPError: If the checkpoint
        download fails.

    References:
    ===========
    [1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
        Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin,
        Jack Clark, Gretchen Krueger, and Ilya Sutskever, "Learning
        Transferable Visual Models From Natural Language Supervision," in
        Proc. ICML, PMLR 139, pp. 8748-8763, 2021.
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
        variant: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str | None = None,
        normalize: bool = True,
        similarity_func: str = "cosine",
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__(
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )
        self._build(variant, pretrained, device=device)
        load_vision_weights(
            self._model, fetch_checkpoint(variant, pretrained, cache_dir=cache_dir)
        )

    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> "ClipEmbedder":
        """
        Build an embedder skeleton from a serialised configuration.

        The pretrained checkpoint is deliberately *not* downloaded: the caller
        restores the weights right after from the serialised ``state_dict``,
        which makes the download both wasteful and a needless network
        dependency. The instance is therefore created without running
        :meth:`__init__`, whose contract is to return a ready-to-use embedder
        with the pretrained weights loaded.

        :param config: The ``"config"`` mapping produced by :meth:`to_dict`.
        :return: A randomly initialized embedder matching ``config``.
        :raises ValueError: If ``variant`` or ``pretrained`` is not supported.
        """
        embedder = cls.__new__(cls)
        SerializableImageEmbedder.__init__(embedder)
        embedder._build(
            config["variant"],
            config["pretrained"],
            device=config["device"],
        )
        return embedder

    def _build(
        self,
        variant: str,
        pretrained: str,
        *,
        device: str | None,
    ) -> None:
        """
        Build the image tower and the preprocessing of a checkpoint.

        The architecture, the preprocessing statistics and the activation
        function all follow from the ``variant`` / ``pretrained`` pair, so this
        sets up everything but the weights, which the caller loads.

        :param variant: CLIP variant name.
        :param pretrained: Pretrained tag naming the weights.
        :param device: Device to run the model on, or ``None`` to auto-select.
        :raises ValueError: If ``variant`` or ``pretrained`` is not supported.
        """
        self._config = get_model_config(variant)
        self._spec = get_checkpoint_spec(variant, pretrained)
        self._variant = variant.replace("/", "-")
        self._pretrained = pretrained
        self._device = resolve_device(device)
        model = build_vision_model(self._config, quick_gelu=self._spec.quick_gelu)
        self._model = model.eval().to(self._device)
        self._transform = _build_preprocess(self._config, self._spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": _CLIP_EMBEDDER_FORMAT_VERSION,
            "embedder_class": type(self).__name__,
            "similarity_func": self._similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "config": {
                "variant": self._variant,
                "pretrained": self._pretrained,
                "device": self._device,
            },
            "state_dict": encode_state_dict(self._model),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "ClipEmbedder":
        cls._reject_unsupported_kwargs(kwargs)
        embedder = cls._from_config(state["config"])
        embedder.similarity_func = state["similarity_func"]
        embedder._restore_normalize(state)
        embedder._restore_batch_size(state)
        embedder._model.load_state_dict(decode_state_dict(state["state_dict"]))
        return embedder

    @property
    def variant(self) -> str:
        """The CLIP variant name in open_clip spelling (e.g. ``"ViT-B-32"``)."""
        return self._variant

    @property
    def pretrained(self) -> str:
        """The pretrained tag naming the loaded weights (e.g. ``"openai"``)."""
        return self._pretrained

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings produced by :meth:`embed`."""
        return self._config.embed_dim

    @property
    def image_size(self) -> int:
        """Side length in pixels of the square model input."""
        return self._config.image_size

    def _preprocess(self, image: UInt8NumpyArray) -> "torch.Tensor":
        """
        Convert one canonical ``uint8`` image into a model-ready tensor.

        The image is routed through PIL and converted to RGB, so grayscale
        ``(H, W)`` inputs are accepted as well as ``(H, W, C)`` ones.

        :param image: A ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
        :return: A normalized image tensor of shape
            ``(3, image_size, image_size)``.
        """
        pil_image = Image.fromarray(image).convert("RGB")
        return cast(torch.Tensor, self._transform(pil_image))

    @torch.no_grad()
    def _embed_batch(self, batch: list[UInt8NumpyArray]) -> Float32NumpyArray:
        """
        Runs one batch of canonical images through the image tower.

        :param batch: Canonical ``uint8`` images of shape ``(H, W[, C])``.
        :return: The ``(len(batch), embedding_dim)`` embeddings.
        """
        tensors = torch.stack([self._preprocess(image) for image in batch])
        features = self._model(tensors.to(self._device))
        return np.asarray(features.float().cpu().numpy(), dtype=np.float32)

    def _embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        embeddings = [
            self._embed_batch(batch)
            for batch in iter_image_batches(
                images, self.batch_size, dims=dims, value_range=value_range
            )
        ]
        if not embeddings:
            raise ValueError("Expected at least one image to embed, got none.")
        return np.vstack(embeddings)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(variant={self._variant}, "
            f"pretrained={self._pretrained}, device={self._device}, "
            f"normalize={self.normalize}, "
            f"similarity_func={self._similarity_func_name})"
        )
