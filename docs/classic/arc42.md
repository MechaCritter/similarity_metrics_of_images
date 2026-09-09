# arc42: classic

Software architecture of `pyvisim.classic`. This document is for developers and
is not part of the published documentation.

## Building block view

An embedder in this module is a local-descriptor aggregator built from three
replaceable parts:

- a **feature extractor** producing the local descriptors, defaulting to
  `RootSIFT`,
- an optional **PCA**, fitted over the descriptors before the clustering model
  sees them,
- a **clustering model** whose fitted parameters are the vocabulary the
  embedding is computed against: `KMeans` for `VLADEmbedder`, a Gaussian
  Mixture Model for `FisherVectorEmbedder`.

`learn` fits these in order: the PCA first, if there is one, then the
clustering model on the projected descriptors. `Pipeline` composes several
fitted embedders by concatenating their vectors, so it owns no vocabulary of
its own.

The clustering models are internal (`pyvisim.classic._clustering`), so their
parameters reach a caller only as the `pca_params`, `kmeans_params` and
`gmm_params` dictionaries the embedders forward.
