Neural networks
===============

This module includes neural networks (``torch.nn.Module``, or neural network -
based - embedders) that learns to distinguish between images.

Table of contents
-----------------

- :doc:`Backbones <backbones/backbones>`
- :doc:`Contrastive Siamese Network <contrastive_siamese/contrastive_siamese>`
- :doc:`BCE Siamese Network <bce_siamese/bce_siamese>`
- :doc:`Triplet Neural Network <triplet/triplet>`
- :doc:`CLIP Embedder <clip/clip>`

.. toctree::
   :maxdepth: 1
   :hidden:

   contrastive_siamese/contrastive_siamese
   bce_siamese/bce_siamese
   triplet/triplet
   clip/clip
   backbones/backbones

Serialization
-------------

Every class in this module can be serialized via method ``to_dict`` and
deserialized via ``from_dict``, or ``save_to_disk`` and ``load_from_disk`` to
save/load to/from a file. ``pyvisim`` only uses ``safetensors`` format for
(de)serialization. See ``docs/arc42.md`` for why.

You can, of course, also use ``torch.save`` and ``torch.load`` as
conventionally used in PyTorch.

References
----------

1. **Siamese Neural Networks for One-shot Image Recognition** (Koch, Zemel, &
   Salakhutdinov, 2015) https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

2. **Dimensionality Reduction by Learning an Invariant Mapping** (Hadsell,
   Chopra, & LeCun, 2006)
   http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

3. **Deep Metric Learning Using Triplet Network** (Hoffer & Ailon, 2014)
   https://arxiv.org/abs/1412.6622

4. **FaceNet: A Unified Embedding for Face Recognition and Clustering**
   (Schroff, Kalenichenko, & Philbin, 2015)
   https://doi.org/10.1109/CVPR.2015.7298682

5. **Deep Residual Learning for Image Recognition**
   https://arxiv.org/abs/1512.03385

6. **Learning Transferable Visual Models From Natural Language Supervision**
   (Radford et al., 2021) https://arxiv.org/abs/2103.00020
