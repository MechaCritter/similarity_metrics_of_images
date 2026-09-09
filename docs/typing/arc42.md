# arc42: typing

Software architecture of `pyvisim.typing`. This document is for developers and
is not part of the published documentation.

## Building block view

The module holds the input types and the normalization helpers every public
method accepts, plus the protocols the library's own components are written
against. The implementation is split: numeric types and image normalization
live in `pyvisim/typing/numeric.py`, the embedder protocol in
`pyvisim/typing/embedders.py`.

## Architecture decisions

### Components are coupled through protocols, not base classes

`Embedder`, `EmbeddingStore` and `SearchIndex` are `typing.Protocol` types, so
a class satisfies one by having the right methods rather than by inheriting
from it. `VLADEmbedder`, `FisherVectorEmbedder` and `Pipeline` satisfy
`Embedder` without any of them importing it, which is what lets
`InMemoryImageEmbeddingStore` accept any of them without importing the concrete
embedder classes, and lets `top_k_map` and `top_k_accuracy` stay decoupled from
the concrete store.

### Every input is normalized to one canonical image once per call

Whatever a caller passes, it is converted to a `uint8` array in `[0, 255]` with
the axes read off the `dims` string, once per call and before the feature
extractor sees it. That single conversion point is why `dims` and
`value_range` appear on every method that takes image data instead of each
component inventing its own layout convention.
