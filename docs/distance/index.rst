Distance metrics
================

Implementations of the pairwise metrics used to
compare image embeddings. Each function takes two 2-D matrices of shape
``(N, D)`` and ``(M, D)`` and returns the full ``(N, M)`` pairwise result.

These are the implementations behind the ``similarity_func`` names
(``"cosine"``, ``"euclidean"``, ``"l1"``, ``"manhattan"``) accepted by the
embedders.

``pyvisim.distance``
--------------------

.. automodule:: pyvisim.distance
   :members:
   :show-inheritance:
