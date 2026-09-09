"""Gallery, query and scoring helpers shared by the end-to-end retrieval tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import Candidate, InMemoryImageEmbeddingStore
from pyvisim.neural_networks import ClipEmbedder
from pyvisim.typing import UInt8NumpyArray

#: CLIP variant and pretrained tag the gallery and the queries are embedded with.
VARIANT = "ViT-B-32"
PRETRAINED = "openai"

#: Images the embedder puts through the image tower in one forward pass.
BATCH_SIZE = 64

#: Graph the gallery embeddings are searched through.
INDEX_PARAMS = {"graph_degree": 16, "build_candidates": 200, "search_candidates": 256}

#: Number of results per query the mean average precision is computed over.
DEPTH = 100

#: Number of candidates retrieved per query before the re-ranking, which needs
#: a pool larger than the results it finally keeps.
POOL = 200

#: Query images decoded at once, which bounds the memory one scoring pass needs.
CHUNK_SIZE = 128

#: Turns a batch of query images into one ranking per query.
Ranker = Callable[[list[UInt8NumpyArray]], list[list[Candidate]]]


@dataclass(frozen=True)
class Gallery:
    """
    The embedded gallery split and the labels its results are scored against.

    :param store: The store holding the gallery embeddings.
    :param path_labels: The flower category of every gallery image path.
    :param class_sizes: Number of gallery images of every flower category.
    """

    store: InMemoryImageEmbeddingStore
    path_labels: dict[str, int]
    class_sizes: Counter[int]


def build_gallery(split: str) -> Gallery:
    """Embed a whole dataset split into an HNSW-backed store.

    :param split: Name of the split the gallery is built from.
    :returns: the store and the labels its results are scored against.
    """
    dataset = OxfordFlowerDataset(purpose=split)
    embedder = ClipEmbedder(VARIANT, PRETRAINED, batch_size=BATCH_SIZE)
    store = InMemoryImageEmbeddingStore(
        dataset.image_paths, embedder, "hnsw", index_params=INDEX_PARAMS
    )
    path_labels = dict(zip(dataset.image_paths, dataset.labels, strict=True))
    return Gallery(store, path_labels, Counter(path_labels.values()))


def mean_average_precision(
    gallery: Gallery, queries: OxfordFlowerDataset, rank: Ranker
) -> float:
    """Score a retrieval configuration over a whole query split.

    :param gallery: The gallery the queries are ranked against.
    :param queries: The split every image of which is used as a query.
    :param rank: The configuration under test, ranking a batch of query images.
    :returns: the mean average precision over the top :data:`DEPTH` results.
    """
    scores = [
        _average_precision(ranked, label, gallery)
        for images, labels in _iter_chunks(queries)
        for ranked, label in zip(rank(images), labels, strict=True)
    ]
    return float(np.mean(scores))


def _iter_chunks(
    queries: OxfordFlowerDataset,
) -> Iterator[tuple[list[UInt8NumpyArray], list[int]]]:
    """Decode the query split :data:`CHUNK_SIZE` images at a time.

    The whole split decoded at once would hold hundreds of megabytes of pixels
    for the length of a scoring pass, so it is read in chunks instead.

    :param queries: The split every image of which is used as a query.
    :returns: an iterator of ``(images, labels)`` chunks, in dataset order.
    """
    for start in range(0, len(queries), CHUNK_SIZE):
        chunk = [
            queries[index]
            for index in range(start, min(start + CHUNK_SIZE, len(queries)))
        ]
        yield [image for image, _, _ in chunk], [label for _, label, _ in chunk]


def _average_precision(
    ranked: Sequence[Candidate], label: int, gallery: Gallery
) -> float:
    """Score the ranking of a single query.

    A result is relevant when its gallery image shows the query's flower
    category. The recall a query can reach within :data:`DEPTH` results is
    capped by the number of relevant gallery images, so the precision sum is
    normalised by ``min(relevant, DEPTH)``.

    :param ranked: The results of the query, best first.
    :param label: The flower category of the query image.
    :param gallery: The gallery the results were retrieved from.
    :returns: the average precision of the ranking.
    """
    hits = np.array(
        [gallery.path_labels[candidate.path] == label for candidate in ranked],
        dtype=bool,
    )
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(precision[hits].sum() / min(gallery.class_sizes[label], DEPTH))
