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

### The index owns the gallery vectors

The vectors are copied into the index at construction time and read back on
demand, so no second copy is held and the store itself does not keep the
original embeddings. `embeddings` therefore returns whatever the index hands
back, which is not always what it was given: a cosine index stores its vectors
L2-normalised, and a compressed external index returns an approximation, or
cannot reconstruct them at all.

The cost of that ownership differs per index. Brute force holds one copy and
scans it, so its ranking is exact at a cost linear in the gallery size, which
also makes it the baseline an approximate index is measured against. The `hnsw`
graph is an additional structure built over the same vectors, so it uses more
memory than brute force does, and it trades recall against speed through
`build_candidates` and `search_candidates`.

### The gallery is read on worker threads and copied on the consumer

`num_workers` threads decode gallery images while the embedder works on the
previous batch, and `num_prefetch_batches` bounds how far they may run ahead.
The threads stop at the decoded image, and the copy into an array happens on
the consuming thread: decoding releases the GIL and so parallelises, while the
copy holds it for its whole duration, so leaving the copy in the threads would
serialise the decodes behind it.

Reads only ever hide behind the embedder's own work, which is the ceiling on
what more threads can buy.

### Alpha query expansion is off by default

Enabling it costs one extra index search per query plus the decoding of
`expansion_neighbours` gallery vectors, which is a price a caller should opt
into rather than pay silently. The weights are computed on L2-normalised copies
whatever `space` the store was built in, because the expansion is defined on
normalised embeddings ranked by cosine similarity.

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
