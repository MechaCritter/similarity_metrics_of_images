MSSSIM
======

``MS-SSIM`` goes one step further than :doc:`SSIM <../ssim/ssim>` by computing
the SSIM at multiple scales. Hence, images are first downsampled by half (up to
5 times) and the SSIM is computed at each scale, then aggregated. Given two
images x and y, ``MS-SSIM(x, y)`` is defined as:

.. math::

   \text{MS-SSIM}(x, y) = \left[l_M(x, y)\right]^{\alpha_M} \cdot \prod_{j=1}^{M} \left[c_j(x, y)\right]^{\beta_j} \left[s_j(x, y)\right]^{\gamma_j}

where :math:`l_j`, :math:`c_j`, :math:`s_j` are the luminance, contrast, and
structure components at scale :math:`j` (as defined in the ``SSIM`` formula),
and :math:`\alpha_M, \beta_j, \gamma_j` are exponents weighting each scale's
contribution. In ``pyvisim``, the default weights, proposed by Wang et al.
(2003), are used.

Usage
-----

.. code-block:: python

   from pyvisim.structural import MSSSIM

   msssim = MSSSIM(batch_size=16)
   matrix = msssim.similarity_score(gallery, queries)   # (N, M) matrix

.. include:: benchmark.md
   :parser: myst_parser.sphinx_

API reference
-------------

.. autoclass:: pyvisim.structural.MSSSIM
   :members:
   :inherited-members:
   :show-inheritance:
