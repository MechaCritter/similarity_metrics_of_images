Pipeline
========

``Pipeline`` glues several classical embedders into one. It embeds an image
with every member, concatenates the per-member vectors, and compares the
combined vectors with a single similarity function. Zhang et al. (2017), for
example, found that combining VLAD and Fisher Vector embeddings improved
fine-grained image recognition performance by 1.1% - 4.8% on Caltech-UCSD 2011
bird, FGVC-Aircraft, FGVC-Cars and Stanford dogs datasets.

.. code-block:: python

   from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder

   vlad = VLADEmbedder(n_clusters=64)
   fisher = FisherVectorEmbedder(n_components=64)
   for embedder in (vlad, fisher):
       embedder.learn(images)

   pipeline = Pipeline([vlad, fisher], similarity_func="cosine")

   vectors = pipeline.embed(images)     # (num_images, vlad_dim + fisher_dim)

   score = pipeline.similarity_score(image1, image2)

   pipeline.save_to_disk("pipeline.embedder")   # Save the pipeline to disk

   # Load the pipeline from disk
   pipeline = Pipeline.load_from_disk("pipeline.embedder")

References
----------

- Zhang, W., Yan, J., Shi, W. et al. Refining deep convolutional features for
  improving fine-grained image recognition. J Image Video Proc. 2017, 27
  (2017). https://doi.org/10.1186/s13640-017-0176-3

API reference
-------------

.. autoclass:: pyvisim.classic.Pipeline
   :members:
   :inherited-members:
   :show-inheritance:
