"""End-to-end check that the k-reciprocal re-ranking lifts the retrieval mAP.

The whole gallery split of the Oxford Flower dataset is embedded by
:class:`~pyvisim.neural_networks.ClipEmbedder` into an HNSW-backed store, and
every image of the query split is searched against it twice: once plainly, and
once by re-ranking a pool of 200 candidates with
:class:`~pyvisim.image_store.KReciprocalReranker` at the parameters of Zhong et
al. The re-ranking is measured on its own here, without the query expansion, so
that the lift is attributable to it alone. Both passes are scored by mean
average precision over the top 100 results, and the re-ranking has to come out
ahead.

The dataset and the CLIP weights are downloaded on first use and the whole
split is embedded, so this module is marked ``slow``.
"""

from __future__ import annotations

import pytest

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import Candidate, KReciprocalReranker
from pyvisim.typing import UInt8NumpyArray

from ._retrieval import DEPTH, POOL, Gallery, mean_average_precision

#: This module embeds a whole dataset split with downloaded CLIP weights.
pytestmark = pytest.mark.slow

#: Size of the neighbourhoods the k-reciprocal sets are built from, as in Zhong et al.
K1 = 20

#: Size of the neighbourhood of the local query expansion, as in Zhong et al.
K2 = 6

#: Weight of the original distance in the final one, as in Zhong et al.
LAMBDA_VALUE = 0.3


def test_reranker_improves_map(gallery: Gallery, queries: OxfordFlowerDataset) -> None:
    """Re-ranked results score a higher mAP than the retrieved ones."""
    reranker = KReciprocalReranker(
        gallery.store, k1=K1, k2=K2, lambda_value=LAMBDA_VALUE
    )

    def plain(images: list[UInt8NumpyArray]) -> list[list[Candidate]]:
        """Rank the gallery for a batch of queries, without any re-ranking."""
        return gallery.store.retrieve_top_k_similar(images, k=DEPTH)

    def reranked(images: list[UInt8NumpyArray]) -> list[list[Candidate]]:
        """Re-rank the candidate pool of every query of a batch."""
        pools = gallery.store.retrieve_top_k_similar(images, k=POOL)
        return [reranker.rerank(pool, top_k=DEPTH) for pool in pools]

    plain_map = mean_average_precision(gallery, queries, plain)
    reranked_map = mean_average_precision(gallery, queries, reranked)
    assert reranked_map > plain_map
