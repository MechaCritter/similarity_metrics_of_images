"""Every built-in backbone works as a :class:`DeepConvFeature` backbone."""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim.features import DeepConvFeature, feature_extractor_from_dict
from pyvisim.neural_networks.backbones import build_backbone, list_backbones
from pyvisim.typing import UInt8NumpyArray

#: Descriptor shape every built-in backbone produces for a 224x224 input: the
#: number of spatial locations of its last conv layer, and its channel count.
_EXPECTED_SHAPES: dict[str, tuple[int, int]] = {
    "resnet18": (49, 512),
    "resnet34": (49, 512),
    "resnet50": (49, 2048),
    "resnet101": (49, 2048),
    "resnet152": (49, 2048),
    "vgg16": (196, 512),
}


@pytest.fixture
def image() -> UInt8NumpyArray:
    """An image the default transform resizes to 224x224.

    :return: A ``(256, 256, 3)`` uint8 image.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def test_every_built_in_backbone_has_an_expected_shape() -> None:
    """The table above covers the whole backbone registry."""
    assert sorted(_EXPECTED_SHAPES) == list_backbones()


@pytest.mark.parametrize("backbone", list_backbones())
def test_output_dim_is_the_channel_count_of_the_last_conv_layer(backbone: str) -> None:
    """``output_dim`` reports the channels of the hooked layer of each backbone."""
    extractor = DeepConvFeature(
        build_backbone(backbone, pretrained=False), device="cpu"
    )
    assert extractor.output_dim == _EXPECTED_SHAPES[backbone][1]


@pytest.mark.parametrize("backbone", list_backbones())
def test_descriptors_of_one_image(backbone: str, image: UInt8NumpyArray) -> None:
    """Each backbone gives one descriptor per spatial location of its last conv layer."""
    extractor = DeepConvFeature(
        build_backbone(backbone, pretrained=False), device="cpu"
    )
    assert extractor(image).shape == _EXPECTED_SHAPES[backbone]


@pytest.mark.parametrize("backbone", list_backbones())
def test_descriptors_of_a_batch(backbone: str, image: UInt8NumpyArray) -> None:
    """A batch gives the same descriptors per image as a single-image call does."""
    extractor = DeepConvFeature(
        build_backbone(backbone, pretrained=False), device="cpu"
    )
    descriptors = extractor.extract_batch([image, image])
    assert [array.shape for array in descriptors] == [_EXPECTED_SHAPES[backbone]] * 2


def test_a_user_supplied_model_cannot_be_rebuilt() -> None:
    """A model that did not come from the registry has no name to rebuild it from."""
    extractor = DeepConvFeature(
        build_backbone("resnet18", pretrained=False), device="cpu"
    )
    state = extractor.to_dict()
    assert state["config"]["backbone"] is None
    with pytest.raises(ValueError, match="user-supplied model"):
        feature_extractor_from_dict(state)


@pytest.mark.parametrize("backbone", list_backbones())
def test_built_in_backbone_builds_from_its_name(backbone: str) -> None:
    """Every registered name builds an extractor with its ImageNet weights."""
    extractor = DeepConvFeature(backbone=backbone, device="cpu")
    assert extractor.output_dim == _EXPECTED_SHAPES[backbone][1]


def test_a_backbone_built_by_name_round_trips() -> None:
    """A serialised extractor rebuilds its backbone by name, and can be saved again."""
    extractor = DeepConvFeature(backbone="resnet18", layer_index=-2, device="cpu")
    reloaded = feature_extractor_from_dict(extractor.to_dict())
    assert isinstance(reloaded, DeepConvFeature)
    assert reloaded.output_dim == extractor.output_dim
    assert reloaded.selected_layer_name == extractor.selected_layer_name
    assert reloaded.to_dict() == extractor.to_dict()
