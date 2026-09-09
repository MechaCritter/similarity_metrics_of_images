RootSIFT
========

File: ``pyvisim/features/_root_sift.py``

RootSIFT is SIFT with Hellinger-kernel normalization. It is the **default
feature extractor** for both ``VLADEmbedder`` and ``FisherVectorEmbedder``.

After computing standard SIFT descriptors, each descriptor is:

1. L1-normalized (divided by the sum of its elements, plus a small epsilon),
   then
2. element-wise square-rooted.

Comparing these transformed vectors with the Euclidean/dot-product operations
the embedders use is equivalent to comparing the original descriptors under the
Hellinger kernel.

Notes
-----

- ``output_dim`` is ``128``, same as SIFT.
- No keypoints yields an empty ``(0, 128)`` array.

References
----------

- R. Arandjelović and A. Zisserman. "Three things everyone should know to
  improve object retrieval". In: 2012 IEEE Conference on Computer Vision and
  Pattern Recognition. 2012, pp. 2911-2918. doi: 10.1109/CVPR.2012.6248018.

API reference
-------------

.. autoclass:: pyvisim.features.RootSIFT
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
