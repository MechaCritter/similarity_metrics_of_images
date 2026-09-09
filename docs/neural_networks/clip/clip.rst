ClipEmbedder
============

Embeds images with a pretrained CLIP image tower. Embeddings are L2-normalized
by default, which makes the cosine similarity a plain dot product.

.. code-block:: python

   from pyvisim.neural_networks import ClipEmbedder

   embedder = ClipEmbedder("ViT-B-32", pretrained="laion2b_s34b_b79k")
   embeddings = embedder.embed(images)                # (N, 512)
   score = embedder.similarity_score(image1, image2)  # (1, 1) cosine similarity

Supported models
----------------

Variant names and pretrained tags follow `open_clip
<https://github.com/mlfoundations/open_clip>`_. OpenAI-style spellings such as
``"ViT-B/32"`` are accepted as aliases of ``"ViT-B-32"``.

.. list-table::
   :header-rows: 1
   :widths: 25 12 12 51

   * - Variant
     - Embedding dim
     - Input size
     - Pretrained tags
   * - ``RN50``
     - 1024
     - 224x224
     - ``openai``, ``yfcc15m``, ``cc12m``
   * - ``RN50-quickgelu``
     - 1024
     - 224x224
     - ``openai``, ``yfcc15m``, ``cc12m``
   * - ``RN101``
     - 512
     - 224x224
     - ``openai``, ``yfcc15m``
   * - ``RN101-quickgelu``
     - 512
     - 224x224
     - ``openai``, ``yfcc15m``
   * - ``RN50x4``
     - 640
     - 288x288
     - ``openai``
   * - ``RN50x4-quickgelu``
     - 640
     - 288x288
     - ``openai``
   * - ``RN50x16``
     - 768
     - 384x384
     - ``openai``
   * - ``RN50x16-quickgelu``
     - 768
     - 384x384
     - ``openai``
   * - ``RN50x64``
     - 1024
     - 448x448
     - ``openai``
   * - ``RN50x64-quickgelu``
     - 1024
     - 448x448
     - ``openai``
   * - ``ViT-B-32``
     - 512
     - 224x224
     - ``openai``, ``laion400m_e31``, ``laion400m_e32``, ``laion2b_e16``, ``laion2b_s34b_b79k``, ``datacomp_xl_s13b_b90k``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-B-32-quickgelu``
     - 512
     - 224x224
     - ``openai``, ``laion400m_e31``, ``laion400m_e32``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-B-32-256``
     - 512
     - 256x256
     - ``datacomp_s34b_b86k``
   * - ``ViT-B-16``
     - 512
     - 224x224
     - ``openai``, ``laion400m_e31``, ``laion400m_e32``, ``laion2b_s34b_b88k``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-B-16-quickgelu``
     - 512
     - 224x224
     - ``openai``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-B-16-plus-240``
     - 640
     - 240x240
     - ``laion400m_e31``, ``laion400m_e32``
   * - ``ViT-L-14``
     - 768
     - 224x224
     - ``openai``, ``laion400m_e31``, ``laion400m_e32``, ``laion2b_s32b_b82k``, ``commonpool_xl_s13b_b90k``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-L-14-quickgelu``
     - 768
     - 224x224
     - ``openai``, ``metaclip_400m``, ``metaclip_fullcc``
   * - ``ViT-L-14-336``
     - 768
     - 336x336
     - ``openai``
   * - ``ViT-L-14-336-quickgelu``
     - 768
     - 336x336
     - ``openai``
   * - ``ViT-H-14``
     - 1024
     - 224x224
     - ``laion2b_s32b_b79k``, ``metaclip_fullcc``, ``metaclip_altogether``
   * - ``ViT-H-14-quickgelu``
     - 1024
     - 224x224
     - ``metaclip_fullcc``
   * - ``ViT-H-14-worldwide``
     - 1024
     - 224x224
     - ``metaclip2_worldwide``
   * - ``ViT-H-14-worldwide-quickgelu``
     - 1024
     - 224x224
     - ``metaclip2_worldwide``
   * - ``ViT-H-14-worldwide-378``
     - 1024
     - 378x378
     - ``metaclip2_worldwide``
   * - ``ViT-g-14``
     - 1024
     - 224x224
     - ``laion2b_s12b_b42k``, ``laion2b_s34b_b88k``
   * - ``ViT-bigG-14``
     - 1280
     - 224x224
     - ``laion2b_s39b_b160k``, ``metaclip_fullcc``
   * - ``ViT-bigG-14-quickgelu``
     - 1280
     - 224x224
     - ``metaclip_fullcc``
   * - ``ViT-bigG-14-worldwide``
     - 1280
     - 224x224
     - ``metaclip2_worldwide``
   * - ``ViT-bigG-14-worldwide-378``
     - 1280
     - 378x378
     - ``metaclip2_worldwide``

The ``-quickgelu`` names are open_clip spellings kept for compatibility, not
separate architectures: whether the tower uses the QuickGELU activation of the
original OpenAI models or the exact GELU of newer checkpoints is read off the
checkpoint itself. A variant and its ``-quickgelu`` twin therefore build the
same model for every tag they share. The plain name only exists separately
because some of them offer extra tags (for example ``ViT-B-32`` adds the LAION
and DataComp checkpoints).

To print all supported variants and pretrained tags, use these helper
functions:

.. code-block:: python

   from pyvisim.neural_networks.clip import available_pretrained, available_variants

   print(available_variants())              # every supported variant name
   print(available_pretrained("ViT-B-32"))  # every pretrained tag of one variant

API reference
-------------

.. autoclass:: pyvisim.neural_networks.ClipEmbedder
   :members:
   :inherited-members:
   :show-inheritance:
