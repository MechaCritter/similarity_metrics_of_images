Features
========

A feature extractor maps one image to a ``(N, D)`` array of local descriptors.
Embedders consume these descriptors and aggregate them into a fixed-size
vector:

.. code-block:: text

   image -> feature extractor -> local descriptors -> embedder -> embedding

The table below includes feature extractors currently implemented in
``pyvisim``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Object
     - ``output_dim``
     - Notes
   * - :doc:`SIFT <sift/sift>`
     - 128
     - SIFT descriptors
   * - :doc:`RootSIFT <rootsift/rootsift>`
     - 128
     - SIFT with Hellinger normalization (default extractor)
   * - :doc:`DeepConvFeature <deep_conv_feature/deep_conv_feature>`
     - layer channels
     - Neural Network feature maps
   * - :doc:`Lambda <lambda/lambda>`
     - user-defined
     - wraps any custom function

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   sift/sift
   rootsift/rootsift
   deep_conv_feature/deep_conv_feature
   lambda/lambda

Reconstructing feature extractors
---------------------------------

.. autofunction:: pyvisim.features.feature_extractor_from_dict
