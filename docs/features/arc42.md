# arc42: features

Software architecture of `pyvisim.features`. This document is for developers
and is not part of the published documentation.

## Building block view

A feature extractor is one end of the contract the embedders are written
against:

```
image -> feature extractor -> local descriptors -> embedder -> embedding
```

Calling an extractor with a single image returns an `(N, D)` array of local
descriptors. `output_dim` declares `D` ahead of the call, which is what lets an
embedder validate an extractor against its PCA and its clustering model before
a single descriptor is computed. Nothing else is required of an extractor, so
`SIFT`, `RootSIFT`, `DeepConvFeature` and a `Lambda` around an arbitrary
function are interchangeable from an embedder's point of view.

`Lambda` exists because that contract is small enough to satisfy without
subclassing `FeatureExtractorBase`. It is also the reason `output_dim` is a
constructor argument there: an arbitrary function has no inspectable descriptor
size, so the value has to be supplied by the caller.

`feature_extractor_from_dict` rebuilds an extractor from a serialized
description, which is how an embedder restores the extractor it was fitted
with.
