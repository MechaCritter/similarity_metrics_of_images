SSIM
====

``SSIM`` captures the perceptual similarity of two images. It is used, for
example, to test out the quality of image compression or denoising algorithms.

Given two images x and y, ``SSIM(x, y)`` is defined as:

.. math::

   \text{SSIM}(x, y) = \underbrace{\left[\frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}\right]}_{\text{luminance}} \cdot \underbrace{\left[\frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}\right]}_{\text{contrast}} \cdot \underbrace{\left[\frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}\right]}_{\text{structure}}

Usage
-----

.. code-block:: python

   from pyvisim.structural import SSIM

   ssim = SSIM()
   scores = ssim.similarity_score(image1, image2)   # (1, 1) matrix

.. include:: benchmark.md
   :parser: myst_parser.sphinx_

API reference
-------------

.. autoclass:: pyvisim.structural.SSIM
   :members:
   :inherited-members:
   :show-inheritance:
