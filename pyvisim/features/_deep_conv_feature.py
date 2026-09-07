from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

from .._base_classes import FeatureExtractorBase
from .._config import setup_logging
from ..lazy_import import OptionalImport
from ..typing import Float32NumpyArray, MatLike
from ._utils import _check_output_shape, _to_single_image

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

    from ..neural_networks.backbones import build_backbone
    from ..utils.torch_utils import resolve_device

setup_logging()

#: Backbone built when none is given.
_DEFAULT_BACKBONE = "vgg16"


def _resolve_backbone(
    backbone: str | torch.nn.Module | None,
) -> tuple[torch.nn.Module, str | None]:
    """Resolve a ``backbone`` argument into a ``(model, backbone name)`` pair.

    The name is the built-in backbone the model was built from, which is all
    serialization needs to rebuild it, and ``None`` for a user-supplied module.
    """
    final_backbone = _DEFAULT_BACKBONE if backbone is None else backbone
    if isinstance(final_backbone, str):
        return build_backbone(final_backbone), final_backbone
    if isinstance(final_backbone, torch.nn.Module):
        return final_backbone, None
    raise TypeError(
        "backbone must be None, a string naming a built-in backbone, or a "
        f"torch.nn.Module. Got {type(final_backbone)} instead."
    )


class DeepConvFeature(FeatureExtractorBase):
    """
    Extracts convolutional feature maps from a chosen conv layer of a torchvision model.
    It flattens the feature maps into feature descriptors.

    The concepts here were inspired by by the work on `VLAD-DCNN` features for face verification, as
    presented in [1], where VLAD embeddings were computed from deep convolutional features and input into
    a metric learning algorithm in order to distinguish between different people.

    :param backbone: The convolutional backbone to extract features from. It may be:

        * ``None`` (default): builds a torchvision VGG16 with ImageNet weights.
        * A string naming a built-in backbone, e.g. ``"vgg16"``, which builds
          the torchvision model with its ImageNet weights. To list all
          supported backbones, use:

          .. code-block:: python

             from pyvisim.neural_networks.backbones import list_backbones

             list_backbones()

        * A ``torch.nn.Module`` instance: any user-supplied PyTorch model.

        In the paper [1], a VGG-Face model trained on the Imdb-Wiki dataset was
        used with VLAD embedding for younger faces verification.
    :param target_submodule: Optional submodule name to hook into. If None, the whole model is used.
    :param layer_index: Which conv layer to hook (int). Use `list_conv_layers(...)`
                       to see the ordering or use -1 for the last conv layer.
    :param device: 'cpu' or 'cuda'. Where to run the model. Defaults to
                   ``None``, which auto-selects 'cuda' when available, else 'cpu'.
    :param transform: Optional torchvision.transforms.Compose. Default includes `to_tensor`, `resize(224, 224)`,
                        and normalization with ImageNet stats.

    .. deprecated:: 0.4.1
        The ``model`` keyword argument is deprecated; pass the model through
        ``backbone`` instead. When ``model`` is supplied it is used as the
        ``backbone`` (unless ``backbone`` is also given) and a
        :class:`FutureWarning` is emitted.

    References:
    ===========
    [1] Liangliang Wang and Deepu Rajan, "An Image Similarity Descriptor for Classification Tasks," J. Vis. Commun. Image R., vol. 71, pp. 102847, 2020.
    [2] Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng, "Refining Deep Convolutional Features for Improving Fine-Grained Image
    Recognition," EURASIP Journal on Image and Video Processing, 2017.
    """

    def __init__(
        self,
        backbone: str | torch.nn.Module | None = None,
        target_submodule: str | None = None,
        layer_index: int = -1,
        device: str | None = None,
        transform: transforms.Compose = None,
        **kwargs: Any,
    ):
        super().__init__()
        _torch_import.check()
        backbone = self._resolve_deprecated_model(backbone, kwargs)
        # Track which built-in backbone is used, if any: serialising stores its
        # name and rebuilds it from torchvision on load, while a user-supplied
        # model has no name to rebuild it from.
        model, self._backbone_name = _resolve_backbone(backbone)
        self._model: torch.nn.Module
        self._target_submodule = target_submodule
        self.layer_index = layer_index
        self.device = resolve_device(device)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            )

        self.model: torch.nn.Module = model  # Trigger setter
        self._modules: torch.nn.Module = self._get_submodule(target_submodule)
        self._conv_layers = self.list_conv_layers()
        if not self._conv_layers:
            raise ValueError(
                f"No convolutional layers found in model {type(self.model).__name__}."
            )

        self.buffer: torch.Tensor | None = None
        try:
            _, self.selected_layer_name, self.selected_layer_module = self._conv_layers[
                self.layer_index
            ]
            self._logger.info(
                f"Selected layer: {self.selected_layer_name}, {self.selected_layer_module}"
            )
        except IndexError as e:
            info = (
                ""
                if target_submodule is None
                else f" in submodule {type(self._modules).__name__}"
            )
            raise IndexError(
                f"Model {type(self.model).__name__} has only {len(self._conv_layers)} convolutional layers {info}"
                f". Got layer_index={self.layer_index}."
            ) from e
        self._output_dim = self.selected_layer_module.out_channels
        self._register_hook()

    @staticmethod
    def _resolve_deprecated_model(
        backbone: str | torch.nn.Module | None,
        kwargs: dict[str, Any],
    ) -> str | torch.nn.Module | None:
        """
        Resolve the deprecated ``model`` keyword argument into ``backbone``.

        If ``model`` is present in ``kwargs`` a :class:`FutureWarning` is
        emitted. The popped ``model`` is used as the ``backbone`` only when no
        explicit ``backbone`` was supplied; otherwise ``backbone`` wins.

        :param backbone: The ``backbone`` argument as passed by the caller.
        :param kwargs: Extra keyword arguments captured by ``__init__``.
        :return: The backbone to use.
        :raises TypeError: If ``kwargs`` contains unexpected keyword arguments.
        """
        if "model" in kwargs:
            warnings.warn(
                "The 'model' argument of DeepConvFeature is deprecated and will "
                "be removed in a future release; pass the model through "
                "'backbone' instead.",
                FutureWarning,
                stacklevel=3,
            )
            model = kwargs.pop("model")
            if backbone is None:
                backbone = model
        if kwargs:
            raise TypeError(
                f"DeepConvFeature got unexpected keyword arguments: {sorted(kwargs)}."
            )
        return backbone

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the configuration needed to rebuild this deep feature extractor.

        Only the name of the built-in backbone is stored, and the model is
        rebuilt from torchvision's default weights on load. A user-supplied
        model has no such name, and is stored as ``None``. The custom
        ``transform`` is not serialised; the default transform is used when
        reconstructing.

        :return: A mapping of constructor arguments.
        """
        return {
            "backbone": self._backbone_name,
            "target_submodule": self._target_submodule,
            "layer_index": self.layer_index,
            "device": self.device,
        }

    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> DeepConvFeature:
        """
        Rebuild a :class:`DeepConvFeature` from a serialised configuration.

        The backbone is rebuilt by name, so the reconstructed extractor can be
        serialised again in turn.

        :param config: Mapping produced by :meth:`_serialization_config`.
        :return: A reconstructed deep feature extractor.
        :raises ValueError: If the extractor was built on a user-supplied model,
            or on a backbone this release no longer knows.
        :raises ImportError: If the optional torch dependency is not installed.
        """
        _torch_import.check()
        backbone = config.get("backbone")
        if backbone is None:
            raise ValueError(
                "Cannot automatically rebuild a DeepConvFeature built on a "
                "user-supplied model. Provide 'feature_extractor' explicitly "
                "when loading."
            )
        return cls(
            backbone=backbone,
            target_submodule=config.get("target_submodule"),
            layer_index=config["layer_index"],
            device=resolve_device(config.get("device", "cpu")),
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Currently, only torch.nn.Module is supported. Got {type(model)} instead."
            )
        self._model = model

    def _get_submodule(self, submodule_name: str | None = None) -> torch.nn.Module:
        """
        Retrieves a submodule from a PyTorch model by name.

        :return: The submodule instance.
        """
        if submodule_name is None:
            return self._model
        if not hasattr(self._model, submodule_name):
            raise AttributeError(
                f"Model {type(self.model).__name__} has no submodule named {submodule_name}."
            )
        submodule = getattr(self._model, submodule_name)
        if not isinstance(submodule, torch.nn.Module):
            raise TypeError(
                f"Attribute {submodule_name} of model {type(self.model).__name__} "
                f"is not a torch.nn.Module, got {type(submodule)} instead."
            )
        return submodule

    def list_conv_layers(self) -> list[tuple[int, str, torch.nn.Conv2d]]:
        """
        Utility function to collect convolutional layers (and sub-modules)
        from the model / chosen submodule.

        :return: List of (layer_index, layer_module) for each convolutional layer.
        """
        conv_layers: list[tuple[int, str, torch.nn.Conv2d]] = []
        idx = 0
        for name, module in self._modules.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((idx, name, module))
                idx += 1
        return conv_layers

    def _register_hook(self) -> None:
        """
        Registers a forward hook on the selected convolutional layer
        to capture its output (feature map).
        """

        def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> None:
            self.buffer = (
                output.detach()
            )  # output shape: [batch_size, channels, height, width]

        self.hook = self.selected_layer_module.register_forward_hook(hook_fn)

    def _feature_maps(self, batch: torch.Tensor) -> Float32NumpyArray:
        """
        Runs a preprocessed batch through the model and returns the hooked maps.

        The model only runs so that the forward hook fires; its own output is
        discarded, and the hook detaches what it captures, so the pass needs no
        autograd graph.

        :param batch: Preprocessed image tensor of shape ``(B, C, H, W)``.
        :return: The captured ``(B, C, Hf, Wf)`` feature maps as a NumPy array.
        :raises RuntimeError: If the forward hook captured nothing.
        """
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            self.model(batch.to(self.device))
        if self.buffer is None:
            raise RuntimeError("Forward hook did not capture any features.")
        return cast(Float32NumpyArray, self.buffer.cpu().numpy())

    def _to_descriptors(self, feature_maps: Float32NumpyArray) -> Float32NumpyArray:
        """
        Flattens hooked feature maps into one descriptor per spatial location.

        :param feature_maps: The ``(B, C, Hf, Wf)`` maps captured by the hook.
        :return: A ``(B, Hf * Wf, D)`` array, with ``D`` the channel count.
        """
        n_images, channels = feature_maps.shape[:2]
        descriptors: Float32NumpyArray = feature_maps.reshape(
            n_images, channels, -1
        ).transpose(0, 2, 1)
        return descriptors

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Processes a single image through the chosen conv layer and
        returns flattened feature descriptors.

        The input is normalized to a canonical ``uint8`` ``(H, W, C)`` image
        and then passed through ``self.transform`` (which converts it to a
        tensor in ``[0, 1]``).

        For the batched version, use :meth:`extract_batch` instead.

        :param image: Input image as ``MatLike`` (e.g. a NumPy ``(H, W, C)``
            array or a torch ``(C, H, W)`` tensor; pass ``dims`` accordingly).
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: N x D NumPy array, where N = (H_conv x W_conv) and
                 D = number_of_channels.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        input_tensor = self.transform(image).unsqueeze(0)
        return cast(
            Float32NumpyArray, self._to_descriptors(self._feature_maps(input_tensor))[0]
        )

    def extract_batch(
        self,
        images: Sequence[MatLike],
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> list[Float32NumpyArray]:
        """
        Extracts the descriptors of a whole batch in a single forward pass.

        The default ``transform`` resizes every image to a fixed size, so the
        batch stacks into one tensor. A custom ``transform`` that preserves the
        input size does not, and the images are then extracted one at a time.

        :param images: Batch of images, each a ``MatLike`` (NumPy array, torch
            tensor or array-like).
        :param dims: Axis-label string describing the layout of every image of
            the batch. See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: One ``(Hf * Wf, D)`` descriptor array per input image.
        """
        tensors, tensor_shapes = [], set()
        for image in images:
            single_image = _to_single_image(image, dims=dims, value_range=value_range)
            transformed_image = self.transform(single_image)
            tensors.append(transformed_image)
            tensor_shapes.add(transformed_image.shape)

        if not tensors:
            return []

        if len(tensor_shapes) > 1:
            self._logger.warning(
                "Images have different shapes after transform, which prevents "
                "batch processing. Falling back to single-image extraction."
            )
            return super().extract_batch(images, dims=dims, value_range=value_range)

        return list(self._to_descriptors(self._feature_maps(torch.stack(tensors))))

    def __repr__(self) -> str:
        return (
            f"DeepConvFeature(backbone={type(self.model).__name__}, layer_index={self.layer_index}, "
            f"device={self.device}, "
            f"transform={self.transform}, selected_layer_name={self.selected_layer_name}, "
            f"selected_layer_module={self.selected_layer_module}, output_dim={self.output_dim})"
        )
