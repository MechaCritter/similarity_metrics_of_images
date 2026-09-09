Classic embedders
=================

Before the deep-learning era, VLAD and Fisher Vector were state-of-the-art
methods for image embedding. They work by extracting local descriptors from all
images from the dataset, then train a clustering model on these descriptors
(``KMeans`` for VLAD, ``Gaussian Mixture Model`` for Fisher). The clustering
model's parameters are then used to compute fixed-length embeddings for the
query images.

Typical local descriptors used were ``SIFT``, ``RootSIFT``, or ``SURF``.

``PCA`` is often applied to reduce the dimensionality of the local descriptors
before aggregation, which can sometimes improve performance.

Table of contents
-----------------

- :doc:`Vector of Locally Aggregated Descriptors <vlad/vlad>`
- :doc:`Fisher Vector <fisher_vector/fisher_vector>`
- :doc:`Pipeline <pipeline/pipeline>`

.. toctree::
   :maxdepth: 1
   :hidden:

   vlad/vlad
   fisher_vector/fisher_vector
   pipeline/pipeline
   pca/pca
