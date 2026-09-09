Backbones
=========

A *backbone* is the pretrained convolutional network that turns an image into
features, before any embedding head or descriptor flattening happens.

Supported backbones
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Name
     - Architecture
     - Weights
     - Feature dim
   * - ``resnet18``
     - ResNet-18
     - ImageNet (torchvision default)
     - 512
   * - ``resnet34``
     - ResNet-34
     - ImageNet (torchvision default)
     - 512
   * - ``resnet50``
     - ResNet-50
     - ImageNet (torchvision default)
     - 2048
   * - ``resnet101``
     - ResNet-101
     - ImageNet (torchvision default)
     - 2048
   * - ``resnet152``
     - ResNet-152
     - ImageNet (torchvision default)
     - 2048
   * - ``vgg16``
     - VGG-16
     - ImageNet (torchvision default)
     - 512

To list the names in code:

.. code-block:: python

   from pyvisim.neural_networks.backbones import build_backbone, list_backbones

   print(list_backbones())                  # every supported backbone name
   model = build_backbone("resnet50")       # torchvision model, ImageNet weights

   # architecture without weights
   bare = build_backbone("resnet50", pretrained=False)

Preprocessing
-------------

Every ResNet is served by the ImageNet preprocessing the torchvision weights
were trained with, which can be obtained via ``get_transform``:

.. code-block:: python

   from pyvisim.neural_networks.backbones import get_transform

   transform = get_transform("resnet50")

References
----------

1. **Deep Residual Learning for Image Recognition** (He, Zhang, Ren, & Sun,
   2015) https://arxiv.org/abs/1512.03385

2. **Very Deep Convolutional Networks for Large-Scale Image Recognition**
   (Simonyan & Zisserman, 2014) https://arxiv.org/abs/1409.1556
