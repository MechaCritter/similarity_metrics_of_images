VLADEmbedder
============

VLAD (Vector of Locally Aggregated Descriptors) was introduced by Jegou et al.
in 2010 as a compact alternative to bag-of-words for large-scale image
retrieval. Similar to BoW, VLAD also uses K-Means for the cluster assignment of
local features, but it aggregates the feature descriptors extracted from an
image into a compact representation. Hence, the values of the descriptors
themselves are considered, not just their occurrences.

VLAD computation
----------------

Given an image, the steps to encode it into a VLAD vector involve the following
steps:

1. **Feature Detection and Description:** Extract descriptors from the image
   using a feature detector algorithm, like ``SIFT``, ``RootSIFT``, or ``SURF``.

2. **Dimension Reduction:** Optionally, reduce the dimensionality of the
   descriptors using PCA to reduce memory usage and computation time.

3. **Descriptor Assignment and Aggregation:** Assign each descriptor to the
   nearest cluster center from a predefined set of centers obtained from a
   K-Means clustering model trained on the descriptors of the training set.
   Then, for each cluster center, aggregate the differences between the
   descriptors assigned to that cluster. Let :math:`x_i` be a descriptor and
   :math:`c_k` be the nearest cluster center. The assignment of :math:`x_i` to
   :math:`c_k` is computed as follows:

   .. math::

      V_k = \sum_{x_i \in k} (x_i - c_k)

   Where :math:`V_k` is the aggregated vector for the cluster :math:`c_k`.

4. **Concatenation:** Concatenate all vectors :math:`V_k` across all clusters to
   form the final VLAD vector.

5. **L2 Normalization:** Since two VLAD vectors are typically compared using
   distance metrics such as Euclidean distance or cosine similarity, it is
   essential to normalize both of them to have the same scale.

Usage
-----

.. code-block:: python

   from pyvisim.classic import VLADEmbedder

   vlad = VLADEmbedder(
       n_clusters=256,                  # number of visual words
       kmeans_params={"rng": 0},        # forwarded to  KMeans
       pca_params={"n_components": 64}, # optional, omit for no PCA
   )
   vlad.learn(images)                   # fits the PCA (if any) then K-Means

   embedding = vlad.embed(image)        # Embed image into a VLAD vector

   # Cosine similarity between two images
   similarity = vlad.similarity_score(image1, image2)

   vlad.save_to_disk("vlad.embedder")   # Save the embedder to disk

   # Load the embedder from disk
   vlad = VLADEmbedder.load_from_disk("vlad.embedder")

The resulting vector has shape ``(K * D,)``, where ``K`` is the number of
clusters and ``D`` is the local descriptor dimension (after optional PCA).

K-Means parameters (``kmeans_params``)
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Meaning
   * - ``n_init``
     - ``1``
     - Number of k-means++ seedings to run. The refined codebook with the
       lowest distortion is kept. Raise it for better, more stable
       vocabularies.
   * - ``thresh``
     - ``1e-05``
     - Stops each refinement once the change in distortion drops below this
       (there is no maximum-iteration count).
   * - ``check_finite``
     - ``True``
     - Whether to validate that the input contains only finite numbers. Turn it
       off for a small speed-up.
   * - ``rng``
     - ``None``
     - Seed (``int``) or :class:`numpy.random.Generator` for reproducible
       fitting.

PCA parameters (``pca_params``)
-------------------------------

See :doc:`PCA <../pca/pca>`.

References
----------

- R. Arandjelović and A. Zisserman. "All About VLAD". In: 2013 IEEE Conference
  on Computer Vision and Pattern Recognition. 2013, pp. 1578-1585.
  doi: 10.1109/CVPR.2013.207.
- H. Jégou, M. Douze, C. Schmid and P. Pérez, "Aggregating local descriptors
  into a compact image representation," 2010 IEEE Computer Society Conference
  on Computer Vision and Pattern Recognition, San Francisco, CA, USA, 2010,
  pp. 3304-3311, doi: 10.1109/CVPR.2010.5540039.

API reference
-------------

.. autoclass:: pyvisim.classic.VLADEmbedder
   :members:
   :inherited-members:
   :show-inheritance:
