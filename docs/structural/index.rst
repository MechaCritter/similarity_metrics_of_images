Structural
==========

Contains the metrics :doc:`SSIM <ssim/ssim>` and
:doc:`MSSSIM <ms_ssim/ms_ssim>`, which captures the perceptual similarity of
two images. It is used, for example, to test out the quality of image
compression or denoising algorithms.

.. code-block:: python

   from pyvisim.structural import MSSSIM, SSIM

   ssim = SSIM()
   scores = ssim.similarity_score(image1, image2)       # (1, 1) matrix

   msssim = MSSSIM(batch_size=16)
   matrix = msssim.similarity_score(gallery, queries)   # (N, M) matrix

.. toctree::
   :maxdepth: 1
   :hidden:

   ssim/ssim
   ms_ssim/ms_ssim
