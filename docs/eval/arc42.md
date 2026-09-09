# arc42: eval

Software architecture of `pyvisim.eval`. This document is for developers and is
not part of the published documentation.

## Building block view

The module is a flat set of scoring functions with no state and no classes.

It depends on the `EmbeddingStore` protocol rather than on
`InMemoryImageEmbeddingStore`, so `top_k_map` and `top_k_accuracy` stay
decoupled from the concrete store. See [the typing arc42](../typing/arc42.md)
for that contract.
