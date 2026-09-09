Introduction
============

``pyvisim`` is a computer vision library for computing image similarities using
traditional and deep learning methods.

Overview
--------

.. image:: https://raw.githubusercontent.com/MechaCritter/Python-Visual-Similarity/assets/docs/architecture/image_embeddings.drawio.png
   :alt: Architecture Diagram

The goal of ``pyvisim`` is to become the largest collection of image similarity
metrics, varying from traditional methods like ``PSNR``, ``SSIM``, ``Fisher
Vectors``, and ``VLAD`` to deep learning methods like ``CLIP`` and ``Siamese
Networks``. Then, one can use these for image retrieval and clustering.

Currently, one would need to install numerous libraries just to get all the
metrics mentioned (for example, ``scikit-image`` + ``opencv-python`` for
``Fisher Vectors`` and ``SSIM``, ``open-clip`` for ``CLIP Embedder``).
``pyvisim`` attempts to close this gap by implementing as many metrics as
possible using only ``numpy``, ``scipy`` (for conventional metrics), and
``torch`` (for deep learning metrics), plus making them more user-friendly with
a simple Object-Oriented code design.

Installation
------------

To install the slim version (**without** deep learning features):

.. code-block:: bash

   pip install pyvisim

Additional features include (note: these pull in heavy dependencies like
``torch``):

.. code-block:: bash

   # For deep learning features and the OxfordFlowerDataset
   pip install "pyvisim[nn]"
