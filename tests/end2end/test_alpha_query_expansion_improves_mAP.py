"""End-to-end check that the alpha query expansion lifts the retrieval mAP.

The whole gallery split of the Oxford Flower dataset is embedded by
:class:`~pyvisim.neural_networks.ClipEmbedder` into an HNSW-backed store, and
every image of the query split is searched against it twice: once plainly and
once with the alpha query expansion. Both passes are scored by mean average
precision over the top 100 results, and the expansion has to come out ahead.

The default parameters of Radenovic et al. are adjusted as following:

- number of neighbours averaged reduced from 50 to 5, since some flower
categories have only 60 images in total, and averaging 50 of them pulls the
query away from its own category.
- The exponent of 3.0 is kept as it is.

The dataset and the CLIP weights are downloaded on first use and the whole
split is embedded, so this module is marked ``slow``.
"""

from __future__ import annotations

import pytest

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import Candidate
from pyvisim.typing import UInt8NumpyArray

from ._retrieval import DEPTH, Gallery, mean_average_precision

#: This module embeds a whole dataset split with downloaded CLIP weights.
pytestmark = pytest.mark.slow

#: Exponent applied to the cosine similarity of each match, as in Radenovic et al.
EXPANSION_ALPHA = 3.0

#: Top-ranked gallery images averaged into the expanded query, sized for the
#: flower categories rather than taken from the paper.
EXPANSION_NEIGHBOURS = 5


def test_alpha_query_expansion_improves_map(
    gallery: Gallery, queries: OxfordFlowerDataset
) -> None:
    """Expanded queries score a higher mAP than the plain ones."""

    def plain(images: list[UInt8NumpyArray]) -> list[list[Candidate]]:
        """Rank the gallery for a batch of queries, without any expansion."""
        return gallery.store.retrieve_top_k_similar(images, k=DEPTH)

    def expanded(images: list[UInt8NumpyArray]) -> list[list[Candidate]]:
        """Rank the gallery for a batch of alpha-expanded queries."""
        return gallery.store.retrieve_top_k_similar(
            images,
            k=DEPTH,
            query_expansion=True,
            expansion_alpha=EXPANSION_ALPHA,
            expansion_neighbours=EXPANSION_NEIGHBOURS,
        )

    plain_map = mean_average_precision(gallery, queries, plain)
    expanded_map = mean_average_precision(gallery, queries, expanded)
    assert expanded_map > plain_map
