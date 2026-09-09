## Benchmark: Use the `ClipEmbedder` for retrieval on the Oxford Flower dataset

> [!IMPORTANT]
> The numbers in this section were produced by
> `scripts/benchmark_reranking.py`. Do not edit this section manually.

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
