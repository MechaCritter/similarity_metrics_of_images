# Image Similarity Retrieval

This module contains objects for storing image embeddings, which
allows for image similarity search.

Using the built-in `hnsw` algorithm, the search is accelerated
significantly for large galleries. The `C++` backend further increases the retrieval speed.

Additionally, the
[alpha query expansion](image_store.md#query-expansion)
averages a query with its best matches before the final search, and the
[k-reciprocal re-ranking](reranking.md)` re-orders a pool of
candidates by how much their neighbourhoods agree with the query's, improving
the `mean average precision` of the retrieval.

## Benchmark: Use the `ClipEmbedder` for retrieval on the Oxford Flower dataset

> [!IMPORTANT]
> The numbers in this section were produced by
> [`scripts/benchmark_reranking.py`](../../scripts/benchmark_reranking.py). Do not edit this section manually.

All 6149 images of the Oxford Flower dataset (`train` split) are embedded by
`ClipEmbedder("ViT-B-32", "openai")` into an `InMemoryImageEmbeddingStore` on the
`hnsw` index, and the 1020 images of the `test` split are the queries. A result is
relevant when it shows the query's flower category. Every configuration returns the
top 100 results of a query: recall@k is the share of queries with a relevant image
among the first k results, mAP@100 the mean average precision over the top 100
(normalised by the relevant images a query can reach within them), and MRR the mean
reciprocal rank of the first relevant result. The hyperparameters were picked on the
`validation` split, and the re-ranking works on a pool of the top 200 candidates of
the expanded query. The time per query includes the embedding of the query image on
the GPU.

<!-- benchmark:begin -->
| Configuration | Recall@1 | Recall@5 | mAP@100 | MRR | Time per query |
|---|---|---|---|---|---|
| Plain retrieval | 90.3% | 97.1% | 55.6% | 93.5% | 4.3 ms |
| Alpha query expansion (`expansion_alpha=0`, `expansion_neighbours=5`) | 87.8% | 94.7% | 62.4% | 90.6% | 4.3 ms |
| Alpha query expansion and k-reciprocal re-ranking (`k1=40`, `k2=6`, `lambda_value=0.1`) | 88.8% | 91.8% | 73.1% | 90.5% | 18.7 ms |

<!-- benchmark:end -->

The precision-recall curves interpolate the precision of every query at fixed
recall levels (the best precision at that recall or beyond) and average it over
the queries. The recall is relative to the relevant images a query can reach
within the top 100.

[Benchmark precision-recall curves](https://raw.githubusercontent.com/MechaCritter/Python-Visual-Similarity/assets/docs/neural_networks/clip_retrieval_precision_recall.png)

## References

- https://www.pinecone.io/learn/series/faiss/hnsw/
- F. Radenović, G. Tolias, and O. Chum, "Fine-tuning CNN Image Retrieval with No
  Human Annotation," IEEE Transactions on Pattern Analysis and Machine Intelligence,
  vol. 41, no. 7, pp. 1655-1668, 2019.
- Z. Zhong, L. Zheng, D. Cao, and S. Li, "Re-ranking Person Re-identification with
  k-reciprocal Encoding," in Proc. CVPR, pp. 1318-1327, 2017.
