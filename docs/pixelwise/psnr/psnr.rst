PSNR
====

The peak signal-to-noise ratio relates the largest possible pixel value (the
*peak signal*) to the mean squared error between the two images (the *noise*).
It is mainly used to measure reconstruction or compression quality.

Given two images x and y of ``N`` pixels each, ``PSNR(x, y)`` is defined as:

.. math::

   \text{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} \left(x_i - y_i\right)^2
   \qquad
   \text{PSNR}(x, y) = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}(x, y)}\right)

where :math:`\text{MAX}` is the peak value of the pixel range, 255 for the
canonical ``uint8`` images ``pyvisim`` normalizes every input to. The result is
reported in decibels: it grows as the two images move closer together, and
identical images give ``MSE = 0`` and therefore ``inf``. Because the logarithm
compresses the scale, the usual reading is comparative: 30 dB is better than
25 dB for the same pair of images, and typical values for lossy compression
land between 30 and 50 dB.

Usage
-----

.. code-block:: python

   from pyvisim.pixelwise import PSNR

   psnr = PSNR()
   scores = psnr.similarity_score(image1, image2)       # (1, 1) matrix, in decibels

   batched = PSNR(batch_size=16)
   matrix = batched.similarity_score(gallery, queries)  # (N, M) matrix

Every compared pair must share the same ``(H, W[, C])`` shape. The squared
differences are summed by a compiled OpenMP kernel, and the
``PYVISIM_NUM_THREADS`` environment variable changes its team size (4 by
default). ``batch_size`` bounds how many images enter one kernel call, which
caps the peak memory of very large galleries.

.. include:: benchmark.md
   :parser: myst_parser.sphinx_

API reference
-------------

.. autoclass:: pyvisim.pixelwise.PSNR
   :members:
   :inherited-members:
   :show-inheritance:
