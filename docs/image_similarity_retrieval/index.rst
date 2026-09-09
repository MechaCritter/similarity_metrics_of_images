Image Similarity Retrieval
==========================

This module contains objects for storing image embeddings, which allows for
image similarity search.

Using the built-in ``hnsw`` algorithm, the search is accelerated significantly
for large galleries. The ``C++`` backend further increases the retrieval speed.

Additionally, the :ref:`alpha query expansion <query-expansion>` averages a
query with its best matches before the final search, and the
:doc:`k-reciprocal re-ranking <reranking/reranking>` re-orders a pool of
candidates by how much their neighbourhoods agree with the query's, improving
the ``mean average precision`` of the retrieval.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   image_store/image_store
   external_search_index/external_search_index
   reranking/reranking

References
----------

- https://www.pinecone.io/learn/series/faiss/hnsw/
- F. Radenović, G. Tolias, and O. Chum, "Fine-tuning CNN Image Retrieval with No
  Human Annotation," IEEE Transactions on Pattern Analysis and Machine
  Intelligence, vol. 41, no. 7, pp. 1655-1668, 2019.
- Z. Zhong, L. Zheng, D. Cao, and S. Li, "Re-ranking Person Re-identification
  with k-reciprocal Encoding," in Proc. CVPR, pp. 1318-1327, 2017.
