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

## Architecture decisions

### The PCA solver is picked from the training shape

`svd_solver="auto"` chooses at fit time: `"covariance_eigh"` when
`n_features <= 1000` and `n_samples >= 10 * n_features`, otherwise `"full"`
when `max(n_samples, n_features) <= 500`, otherwise `"arpack"` when
`n_components < 0.8 * min(n_samples, n_features)`, and `"full"` otherwise.

Component signs are made deterministic by flipping each component so that its
largest-magnitude entry is positive, which is what makes the output of the four
solvers comparable.

### Serialization preserves the memory order of fitted arrays

Some fitted attributes are stored Fortran-contiguous, and the matrix-product
code path differs by layout, so rebuilding such an array in C order would not
reproduce the exact same floating-point results. The serializer therefore
records the order and restores it, which is what keeps a round-tripped embedder
bit-for-bit reproducible.
