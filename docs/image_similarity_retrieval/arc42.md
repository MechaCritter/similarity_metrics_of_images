# arc42: image_similarity_retrieval

Software architecture of `pyvisim.image_store`. This document is for developers
and is not part of the published documentation.

## Building block view

A store owns a gallery and a search index. The index owns the gallery vectors,
answers a search with row numbers and scores, and hands individual vectors back
on demand. `InMemoryImageEmbeddingStore` turns those row numbers into
`Candidate` objects and keeps the embedder that produced the gallery, so a
query image can be embedded the same way the gallery was.

Three index implementations sit behind the same interface: the exact brute
force index, the `hnsw` graph, and `ExternalSearchIndex`.

## Architecture decisions

### The index vocabulary is pyvisim's, not the backend's

Index parameters are named after what they do rather than after the library
that implements the index. Each index owns one table mapping those names onto
the keywords its backend actually understands, so a caller's vocabulary stays
put if the backend behind an index ever changes: only the table moves.

### `ExternalSearchIndex` is an adapter, not a dependency

The adapter lets a store search through an index somebody else built, a FAISS
index in particular, without this package depending on the library that
produced it. The consequence is that the scores stay the external index's own:
an L2 index reports distances, an inner-product index reports similarities, and
`pyvisim` cannot tell which metric produced either. Normalisation is therefore
the caller's job, and a lossy index cannot always reconstruct the vectors it
was given, which is why `save_to_disk` accepts them explicitly and
`load_from_disk` takes a rebuilt index back.

### The reranker requires a store on a built-in index

Status: current, revisitable.

The reranker reads the candidates' embeddings back from the store's index to
compute the distances among them in the store's `space`, while the query's
distances to the candidates are the scores the store ranked them by. Both
therefore speak the same metric, which is why the candidates must come from the
given store, and why a store on an `ExternalSearchIndex`, whose scores may be
similarities or distances of an unknown metric, is currently rejected.
