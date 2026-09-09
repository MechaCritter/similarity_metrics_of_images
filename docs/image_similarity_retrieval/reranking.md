# Re-ranking

## k-reciprocal re-ranking

`KReciprocalReranker` re-orders the candidates a store retrieved for a query with
the **k-reciprocal encoding** of Zhong et al. [1]. Two images are k-reciprocal
neighbours when each is among the `k1` nearest neighbours of the other, a far
stricter relation than merely being close to the query: a false match may lie
close to the query, but the query rarely lies close to the false match's own
neighbours. The query and every candidate are encoded into a k-reciprocal
feature, a vector over the candidate set that holds a Gaussian weight for each of
their k-reciprocal neighbours and zero elsewhere, and the Jaccard distance between
the query's feature and a candidate's says how much their neighbourhoods agree.
The candidates are finally re-ranked by
`(1 - lambda_value) * jaccard + lambda_value * original`, where `original` is the
distance the store ranked them by.

### Reranking algorithm

Let the probe $p$ be the query and $\mathcal{G} = \{g_i \mid i = 1, 2, \dots, N\}$ are
the candidates, so the set the neighbourhoods are built over holds $N + 1$
images.

- **Original distance.** $d(p, g_i)$ is the score the store ranked the candidate
  by, and $d(g_i, g_j)$ is computed from the candidates' embeddings in the
  store's `space`. Every row of it is divided by its largest entry (reference
  implementation), and hence lies in range $[0, 1]$.

- **k-nearest neighbours, Eq. (2).** The ranking list $\mathcal{L}(p, \mathcal{G})$
  sorts the set by $d$, and $N(p, k)$ is its top-$k$:

  $$
  N(p, k) = \{g_1^0, g_2^0, \dots, g_k^0\}, \quad |N(p, k)| = k
  $$

  The ranking list runs over the probe and the candidates together, the image
  itself at rank zero (reference implementation), and the same definition serves
  every candidate $g_i$ in place of $p$.

- **k-reciprocal neighbours, Eq. (3).** Only the neighbours that hold $p$ among
  their own $k$ nearest neighbours are kept:

  $$
  \mathcal{R}(p, k) = \{g_i \mid (g_i \in N(p, k)) \wedge (p \in N(g_i, k))\}
  $$

- **Expansion, Eq. (4).** With $k = k_1$, every member $q$ of $\mathcal{R}(p, k)$
  brings its own $\tfrac{1}{2}k$-reciprocal neighbours in, provided at least two
  thirds of them already lie in $\mathcal{R}(p, k)$:

  $$
  \mathcal{R}^*(p, k) \leftarrow \mathcal{R}(p, k) \cup \mathcal{R}(q, \tfrac{1}{2}k)
  \quad \text{s.t.} \quad
  |\mathcal{R}(p, k) \cap \mathcal{R}(q, \tfrac{1}{2}k)| \geq \tfrac{2}{3} |\mathcal{R}(q, \tfrac{1}{2}k)|,
  \quad \forall q \in \mathcal{R}(p, k)
  $$

- **k-reciprocal feature, Eq. (7).** Each image is encoded into a vector over the
  set, a Gaussian kernel of the original distance on its expanded neighbourhood
  and zero elsewhere:

  $$
  \mathcal{V}_{p, g_i} =
  \begin{cases}
  e^{-d(p, g_i)} & \text{if } g_i \in \mathcal{R}^*(p, k_1) \\
  0 & \text{otherwise}
  \end{cases}
  $$

  Each vector is scaled to unit $L_1$ norm (reference implementation), so the
  Jaccard distance below compares the shape of two neighbourhoods rather than
  their size.

- **Local query expansion, Eq. (11).** The feature of every image is replaced by
  the mean feature of its $k_2$ nearest neighbours, the image itself included
  (reference implementation). `k2=1` skips this step.

  $$
  \mathcal{V}_p = \frac{1}{|N(p, k_2)|} \sum_{g_i \in N(p, k_2)} \mathcal{V}_{g_i}
  $$

- **Jaccard distance, Eq. (10).** The overlap of two neighbourhoods is read off
  their features with the element-wise minimum and maximum:

  $$
  d_J(p, g_i) = 1 - \frac{\sum_{j=1}^{N} \min(\mathcal{V}_{p, g_j}, \mathcal{V}_{g_i, g_j})}
  {\sum_{j=1}^{N} \max(\mathcal{V}_{p, g_j}, \mathcal{V}_{g_i, g_j})}
  $$

- **Final distance, Eq. (12).** The Jaccard distance is mixed with the original
  one, and the candidates are sorted by it in ascending order. `rerank` returns
  the first `top_k` of them, each scored by its $d^*$:

  $$
  d^*(p, g_i) = (1 - \lambda) \, d_J(p, g_i) + \lambda \, d(p, g_i)
  $$

## Architecture Decision Record

- The reranker reads the candidates' embeddings back from the store's index to
compute the distances among them in the store's `space`, while the query's
distances to the candidates are the scores the store ranked them by. Both
therefore speak the same metric, which is why the candidates must come from the
given store, and why a store on an `ExternalSearchIndex`, whose scores may be
similarities or distances of an unknown metric, is currently rejected.

## References

[1] Z. Zhong, L. Zheng, D. Cao, and S. Li, "Re-ranking Person Re-identification
with k-reciprocal Encoding," in Proc. CVPR, pp. 1318-1327, 2017.
