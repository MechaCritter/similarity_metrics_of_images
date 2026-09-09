SIFT
====

File: ``pyvisim/features/_sift.py``

Scale-Invariant Feature Transform descriptors. SIFT was the original local
descriptor used for VLAD and Fisher Vector embedding.

- ``output_dim`` is ``128`` (standard SIFT descriptor length).

For most uses prefer :doc:`RootSIFT <../rootsift/rootsift>`, which normalizes
these descriptors and usually improves retrieval at no extra cost.

References
----------

- D. G. Lowe. "Distinctive Image Features from Scale-Invariant Keypoints". In:
  International Journal of Computer Vision 60.2 (2004), pp. 91-110.
  issn: 1573-1405. doi: 10.1023/B:VISI.0000029664.99615.94.
  url: https://doi.org/10.1023/B:VISI.0000029664.99615.94.

API reference
-------------

.. autoclass:: pyvisim.features.SIFT
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
