"""
Measure what query expansion and re-ranking buy the image store.

An ``InMemoryImageEmbeddingStore`` over the Oxford Flower ``train`` split,
embedded by CLIP and searched through an HNSW graph, answers the queries of the
``validation`` split under a grid of hyperparameters, the best setting of each
stage is picked there, and three configurations are finally scored on the
``test`` split: plain retrieval, alpha query expansion, and alpha query expansion
followed by k-reciprocal re-ranking. Recall@1, recall@5, mAP and MRR over the
top ``depth`` results are written to a markdown report together with the sweep
tables, and the precision-recall curves of the three configurations are plotted
to a PNG.

Run it with::

    uv run --extra nn --group bench python scripts/benchmark_reranking.py
"""

import argparse
import os
import platform
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TypeVar

import matplotlib
import numpy as np

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import (
    Candidate,
    InMemoryImageEmbeddingStore,
    KReciprocalReranker,
)
from pyvisim.neural_networks import ClipEmbedder
from pyvisim.typing import FloatNumpyArray, UInt8NumpyArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_REF = "scripts/benchmark_reranking.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The gallery, the split the hyperparameters are picked on, and the split the
#: final numbers are reported on.
_GALLERY_SPLIT = "train"
_TUNING_SPLIT = "validation"
_TEST_SPLIT = "test"

#: The embedder and the index the store is built with.
_VARIANT = "ViT-B-32"
_PRETRAINED = "openai"
_INDEX_PARAMS = {"graph_degree": 16, "build_candidates": 200, "search_candidates": 256}

#: Grid of the alpha query expansion, swept on the tuning split.
_ALPHAS = (0.0, 1.0, 2.0, 3.0, 5.0)
_NEIGHBOURS = (2, 3, 5, 10, 20, 50)
#: Grid of the k-reciprocal re-ranking, swept on top of the selected expansion.
_K1S = (10, 20, 40)
_K2S = (1, 3, 6)
_LAMBDAS = (0.1, 0.3, 0.5)

#: Recall levels the precision-recall curves are interpolated at.
_RECALL_LEVELS = np.linspace(0.0, 1.0, 101)

#: Chart chrome and the three fixed series slots of the validated default
#: dataviz palette: blue = plain, orange = expansion, aqua = expansion plus
#: re-ranking.
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
#: Smallest vertical gap, in precision units, at which direct labels are placed
#: beside the curves instead of being left to the legend.
_MIN_LABEL_GAP = 0.04

_Rankings = list[list[Candidate]]
#: A hyperparameter setting of either stage.
_Setting = TypeVar("_Setting")


def _log(message: str) -> None:
    """Print a progress line right away, so a long run can be followed."""
    print(message, flush=True)


@dataclass(frozen=True)
class Split:
    """
    The query images of one dataset split, decoded and labelled.

    :param name: Name of the split.
    :param images: The decoded RGB query images.
    :param labels: The class label of every query image.
    """

    name: str
    images: list[UInt8NumpyArray]
    labels: list[int]


@dataclass(frozen=True)
class ExpansionSetting:
    """
    One setting of the alpha query expansion.

    :param alpha: Exponent of the similarity weights.
    :param neighbours: Top-ranked gallery images averaged into the query.
    """

    alpha: float
    neighbours: int

    def describe(self) -> str:
        """Name the setting for the report."""
        return f"alpha={self.alpha:g}, {self.neighbours} neighbours"


@dataclass(frozen=True)
class RerankSetting:
    """
    One setting of the k-reciprocal re-ranking.

    :param k1: Size of the neighbourhoods the k-reciprocal sets are built from.
    :param k2: Size of the neighbourhood of the local query expansion.
    :param lambda_value: Weight of the original distance in the final one.
    """

    k1: int
    k2: int
    lambda_value: float

    def describe(self) -> str:
        """Name the setting for the report."""
        return f"k1={self.k1}, k2={self.k2}, lambda={self.lambda_value:g}"


@dataclass(frozen=True)
class Scores:
    """
    The retrieval quality of one configuration over a set of queries.

    :param recall_at_1: Share of queries whose best match is relevant.
    :param recall_at_5: Share of queries with a relevant match among the top 5.
    :param mean_average_precision: Mean average precision over the top results.
    :param mean_reciprocal_rank: Mean of ``1 / rank`` of the first relevant
        match, ``0`` when there is none.
    :param precision_at_recall: Interpolated precision at every recall level of
        :data:`_RECALL_LEVELS`, averaged over the queries.
    :param seconds_per_query: Wall-clock time the configuration took per query,
        the embedding of the query included.
    """

    recall_at_1: float
    recall_at_5: float
    mean_average_precision: float
    mean_reciprocal_rank: float
    precision_at_recall: FloatNumpyArray
    seconds_per_query: float


def _load_split(name: str, num_queries: int | None, seed: int) -> Split:
    """
    Decode the query images of one split.

    :param name: Name of the split.
    :param num_queries: Number of queries to sample, or ``None`` for all.
    :param seed: Seed of the sampling.
    :return: The decoded queries and their labels.
    :raises ValueError: If the split holds fewer images than requested.
    """
    dataset = OxfordFlowerDataset(purpose=name)
    if num_queries is None:
        indices = np.arange(len(dataset))
    elif num_queries > len(dataset):
        raise ValueError(
            f"The {name!r} split holds {len(dataset)} images, but {num_queries} "
            f"were requested."
        )
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(dataset), size=num_queries, replace=False))
    images: list[UInt8NumpyArray] = []
    labels: list[int] = []
    for index in indices:
        image, label, _ = dataset[int(index)]
        images.append(image)
        labels.append(label)
    return Split(name, images, labels)


def _build_store(
    batch_size: int,
) -> tuple[InMemoryImageEmbeddingStore, dict[str, int]]:
    """
    Embed the gallery split into an HNSW-backed store.

    :param batch_size: Batch size of the CLIP embedder.
    :return: The store and the class label of every gallery path.
    """
    dataset = OxfordFlowerDataset(purpose=_GALLERY_SPLIT)
    embedder = ClipEmbedder(_VARIANT, _PRETRAINED, batch_size=batch_size)
    _log(f"Embedding {len(dataset)} gallery images on {embedder.device}.")
    start = time.perf_counter()
    store = InMemoryImageEmbeddingStore(
        dataset.image_paths, embedder, "hnsw", index_params=_INDEX_PARAMS
    )
    _log(f"Built the store in {time.perf_counter() - start:.0f} s.")
    return store, dict(zip(dataset.image_paths, dataset.labels, strict=True))


def _retrieve(
    store: InMemoryImageEmbeddingStore,
    split: Split,
    depth: int,
    expansion: ExpansionSetting | None,
) -> tuple[_Rankings, float]:
    """
    Rank the gallery for every query of a split.

    :param store: The store to search.
    :param split: The queries.
    :param depth: Number of results per query.
    :param expansion: The alpha query expansion to apply, or ``None``.
    :return: The rankings and the wall-clock seconds the pass took.
    """
    start = time.perf_counter()
    if expansion is None:
        rankings = store.retrieve_top_k_similar(split.images, k=depth)
    else:
        rankings = store.retrieve_top_k_similar(
            split.images,
            k=depth,
            query_expansion=True,
            expansion_alpha=expansion.alpha,
            expansion_neighbours=expansion.neighbours,
        )
    return rankings, time.perf_counter() - start


def _rerank(
    reranker: KReciprocalReranker, pools: _Rankings, depth: int
) -> tuple[_Rankings, float]:
    """
    Re-rank every candidate pool and keep the best ``depth`` of each.

    :param reranker: The reranker.
    :param pools: The candidate pool of every query.
    :param depth: Number of results kept per query.
    :return: The re-ranked results and the wall-clock seconds the pass took.
    """
    start = time.perf_counter()
    rankings = [reranker.rerank(pool, top_k=depth) for pool in pools]
    return rankings, time.perf_counter() - start


def _interpolated_precision(
    precision: FloatNumpyArray, recall: FloatNumpyArray
) -> FloatNumpyArray:
    """
    Interpolate a precision-recall curve at the fixed recall levels.

    :param precision: Precision after every rank.
    :param recall: Recall after every rank.
    :return: At every level, the best precision at that recall or beyond, ``0``
        where the recall is never reached.
    """
    return np.array(
        [precision[recall >= level].max(initial=0.0) for level in _RECALL_LEVELS]
    )


def _score(
    rankings: _Rankings,
    split: Split,
    path_labels: dict[str, int],
    class_sizes: Counter[int],
    depth: int,
    seconds: float,
) -> Scores:
    """
    Score the rankings of every query of a split.

    A result is relevant when its gallery image carries the query's class. The
    recall a query can reach within ``depth`` results is capped by the number
    of relevant gallery images, so both the average precision and the recall
    axis of the curve are normalised by ``min(relevant, depth)``.

    :param rankings: The ranked results of every query.
    :param split: The queries.
    :param path_labels: The class label of every gallery path.
    :param class_sizes: The number of gallery images of every class.
    :param depth: Number of results scored per query.
    :param seconds: Wall-clock seconds the configuration took for the split.
    :return: The scores, averaged over the queries.
    """
    first_hits: list[float] = []
    top_five_hits: list[float] = []
    average_precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    curves: list[FloatNumpyArray] = []
    for ranked, label in zip(rankings, split.labels, strict=True):
        hits = np.array([path_labels[c.path] == label for c in ranked], dtype=bool)
        reachable = min(class_sizes[label], depth)
        found = np.cumsum(hits)
        precision = found / np.arange(1, hits.size + 1)
        recall = found / reachable
        relevant_ranks = np.flatnonzero(hits)
        first_hits.append(float(hits[0]))
        top_five_hits.append(float(hits[:5].any()))
        average_precisions.append(float(precision[hits].sum() / reachable))
        reciprocal_ranks.append(
            1.0 / (relevant_ranks[0] + 1) if relevant_ranks.size else 0.0
        )
        curves.append(_interpolated_precision(precision, recall))
    return Scores(
        recall_at_1=float(np.mean(first_hits)),
        recall_at_5=float(np.mean(top_five_hits)),
        mean_average_precision=float(np.mean(average_precisions)),
        mean_reciprocal_rank=float(np.mean(reciprocal_ranks)),
        precision_at_recall=np.mean(curves, axis=0),
        seconds_per_query=seconds / len(split.images),
    )


def _sweep_expansion(
    store: InMemoryImageEmbeddingStore,
    split: Split,
    path_labels: dict[str, int],
    class_sizes: Counter[int],
    depth: int,
) -> dict[ExpansionSetting, Scores]:
    """
    Score every setting of the alpha query expansion on a split.

    :param store: The store to search.
    :param split: The queries.
    :param path_labels: The class label of every gallery path.
    :param class_sizes: The number of gallery images of every class.
    :param depth: Number of results scored per query.
    :return: The scores of every setting, in grid order.
    """
    scores: dict[ExpansionSetting, Scores] = {}
    for alpha in _ALPHAS:
        for neighbours in _NEIGHBOURS:
            setting = ExpansionSetting(alpha, neighbours)
            rankings, seconds = _retrieve(store, split, depth, setting)
            scores[setting] = _score(
                rankings, split, path_labels, class_sizes, depth, seconds
            )
            _log(
                f"[{split.name}] expansion {setting.describe()}: "
                f"mAP@{depth} {scores[setting].mean_average_precision:.4f}"
            )
    return scores


def _sweep_reranking(
    store: InMemoryImageEmbeddingStore,
    pools: _Rankings,
    split: Split,
    path_labels: dict[str, int],
    class_sizes: Counter[int],
    depth: int,
) -> dict[RerankSetting, Scores]:
    """
    Score every setting of the k-reciprocal re-ranking on fixed candidate pools.

    :param store: The store the pools were retrieved from.
    :param pools: The candidate pool of every query.
    :param split: The queries.
    :param path_labels: The class label of every gallery path.
    :param class_sizes: The number of gallery images of every class.
    :param depth: Number of results kept and scored per query.
    :return: The scores of every setting, in grid order.
    """
    scores: dict[RerankSetting, Scores] = {}
    for k1 in _K1S:
        for k2 in _K2S:
            for lambda_value in _LAMBDAS:
                setting = RerankSetting(k1, k2, lambda_value)
                reranker = KReciprocalReranker(
                    store, k1=k1, k2=k2, lambda_value=lambda_value
                )
                rankings, seconds = _rerank(reranker, pools, depth)
                scores[setting] = _score(
                    rankings, split, path_labels, class_sizes, depth, seconds
                )
                _log(
                    f"[{split.name}] re-ranking {setting.describe()}: "
                    f"mAP@{depth} {scores[setting].mean_average_precision:.4f}"
                )
    return scores


def _best(scores: dict[_Setting, Scores]) -> _Setting:
    """Return the setting with the highest mean average precision."""
    return max(scores, key=lambda setting: scores[setting].mean_average_precision)


def _plot_precision_recall(
    curves: dict[str, FloatNumpyArray], num_queries: int, depth: int, path: Path
) -> None:
    """
    Plot the precision-recall curves of the configurations.

    :param curves: The interpolated precision of every configuration at the
        recall levels of :data:`_RECALL_LEVELS`, in the order of the series
        colours.
    :param num_queries: Number of queries the curves average over.
    :param depth: Number of results the curves were computed over.
    :param path: PNG file the figure is written to.
    """
    figure, axes = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    figure.patch.set_facecolor(_SURFACE)
    axes.set_facecolor(_SURFACE)
    for (label, curve), color in zip(curves.items(), _SERIES_COLORS, strict=True):
        axes.plot(
            _RECALL_LEVELS,
            curve,
            color=color,
            linewidth=2.0,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=label,
        )
    _place_direct_labels(axes, curves)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.02)
    axes.set_xlabel(
        f"Recall of the relevant images reachable in the top {depth}",
        color=_INK_SECONDARY,
    )
    axes.set_ylabel("Precision", color=_INK_SECONDARY)
    axes.set_title(
        f"Precision-recall on the Oxford Flower dataset, {num_queries} test queries",
        color=_INK,
        loc="left",
        fontsize=11,
    )
    axes.grid(True, color=_GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_AXIS)
    axes.tick_params(colors=_MUTED, labelsize=9)
    legend = axes.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_INK_SECONDARY)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=_SURFACE)
    plt.close(figure)


def _place_direct_labels(axes: plt.Axes, curves: dict[str, FloatNumpyArray]) -> None:
    """
    Label the curves where they lie furthest apart, if they lie apart at all.

    The label of every curve is set beside the recall level at which the
    curves spread the most, on the side of the plot with the most room. When
    two of them would still sit closer than :data:`_MIN_LABEL_GAP` there, the
    labels are left out and the legend alone carries the identity, so they
    never collide.

    :param axes: The axes the curves are drawn on.
    :param curves: The curves, by name.
    """
    stacked = np.stack(list(curves.values()))
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    level = int(np.argmax(spread))
    heights = np.sort(stacked[:, level])
    if len(heights) > 1 and np.diff(heights).min() < _MIN_LABEL_GAP:
        return
    on_the_right = _RECALL_LEVELS[level] < 0.5
    for name, curve in curves.items():
        axes.annotate(
            name,
            (_RECALL_LEVELS[level], curve[level]),
            xytext=(6 if on_the_right else -6, 0),
            textcoords="offset points",
            color=_INK_SECONDARY,
            fontsize=8,
            ha="left" if on_the_right else "right",
            va="center",
        )


def _format_percent(value: float) -> str:
    """Format a share as a percentage with one decimal."""
    return f"{100.0 * value:.1f}%"


def _format_results_table(rows: Sequence[tuple[str, Scores]], depth: int) -> str:
    """Render the headline numbers of the configurations as a markdown table."""
    lines = [
        f"| Configuration | Recall@1 | Recall@5 | mAP@{depth} | MRR | Time per query (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for label, scores in rows:
        lines.append(
            f"| {label} | {_format_percent(scores.recall_at_1)} "
            f"| {_format_percent(scores.recall_at_5)} "
            f"| {_format_percent(scores.mean_average_precision)} "
            f"| {_format_percent(scores.mean_reciprocal_rank)} "
            f"| {1000.0 * scores.seconds_per_query:.1f} |"
        )
    return "\n".join(lines)


def _format_expansion_sweep(
    scores: dict[ExpansionSetting, Scores], best: ExpansionSetting
) -> str:
    """Render the expansion sweep as a grid of mAP values, the best in bold."""
    header = " | ".join(f"{n} neighbours" for n in _NEIGHBOURS)
    lines = [f"| alpha | {header} |", "|---|" + "---|" * len(_NEIGHBOURS)]
    for alpha in _ALPHAS:
        cells = []
        for neighbours in _NEIGHBOURS:
            setting = ExpansionSetting(alpha, neighbours)
            cell = _format_percent(scores[setting].mean_average_precision)
            cells.append(f"**{cell}**" if setting == best else cell)
        lines.append(f"| {alpha:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_reranking_sweep(
    scores: dict[RerankSetting, Scores], best: RerankSetting, depth: int
) -> str:
    """Render the re-ranking sweep as a table, the best setting in bold."""
    lines = [
        f"| k1 | k2 | lambda | Recall@1 | mAP@{depth} | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for setting, result in scores.items():
        cells = [
            str(setting.k1),
            str(setting.k2),
            f"{setting.lambda_value:g}",
            _format_percent(result.recall_at_1),
            _format_percent(result.mean_average_precision),
            _format_percent(result.mean_reciprocal_rank),
        ]
        if setting == best:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _package_version(name: str) -> str:
    """Resolve an installed package version, or ``"unknown"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_report(
    results: Sequence[tuple[str, Scores]],
    expansion_scores: dict[ExpansionSetting, Scores],
    best_expansion: ExpansionSetting,
    reranking_scores: dict[RerankSetting, Scores],
    best_reranking: RerankSetting,
    plain_tuning: Scores,
    gallery_size: int,
    num_tuning: int,
    num_test: int,
    depth: int,
    pool: int,
    device: str,
    plot: Path,
    output: Path,
) -> str:
    """Render the complete markdown report."""
    plot_link = Path(os.path.relpath(plot, output.parent)).as_posix()
    return f"""# Query expansion and re-ranking benchmark

> [!IMPORTANT]
> This file was generated by the script [`{_SCRIPT_REF}`](../../{_SCRIPT_REF}).
> **Do not edit manually!**

All {gallery_size} images of the Oxford Flower dataset (`{_GALLERY_SPLIT}` split)
are embedded by `ClipEmbedder("{_VARIANT}", "{_PRETRAINED}")` on the `{device}`
device into an `InMemoryImageEmbeddingStore` on the `hnsw` index with
`{_INDEX_PARAMS}`. A result is relevant when its gallery image shows the same
flower category as the query. Every configuration returns the top {depth}
results of a query, over which the metrics are computed:

- **Recall@k**: share of queries with at least one relevant image among the
  first k results.
- **mAP@{depth}**: mean average precision over the top {depth} results, normalised
  by the number of relevant images a query can reach within them.
- **MRR**: mean reciprocal rank of the first relevant result, 0 when there is
  none in the top {depth}.
- **Time per query**: wall-clock time of the whole configuration divided by the
  number of queries, the embedding of the query image included.

The hyperparameters are picked on the {num_tuning} queries of the
`{_TUNING_SPLIT}` split by mAP@{depth}, the expansion first and the re-ranking on
top of the selected expansion, and the final numbers are reported on the
{num_test} queries of the `{_TEST_SPLIT}` split, which take no part in the
selection. The re-ranking works on a pool of the top {pool} candidates of the
expanded query and keeps the best {depth} of them.

## Results ({_TEST_SPLIT} split)

{_format_results_table(results, depth)}

![Precision-recall curves]({plot_link})

The precision-recall curves interpolate the precision of every query at fixed
recall levels (the best precision at that recall or beyond) and average it over
the queries. The recall is relative to the relevant images a query can reach
within the top {depth}.

## Hyperparameter search ({_TUNING_SPLIT} split)

Plain retrieval scores {_format_percent(plain_tuning.mean_average_precision)}
mAP@{depth} on this split.

### Alpha query expansion (mAP@{depth})

{_format_expansion_sweep(expansion_scores, best_expansion)}

### k-reciprocal re-ranking, on top of the expansion with {best_expansion.describe()}

{_format_reranking_sweep(reranking_scores, best_reranking, depth)}

## Environment

| | |
|---|---|
| pyvisim | {_package_version("pyvisim")} |
| Python | {platform.python_version()} |
| NumPy | {_package_version("numpy")} |
| PyTorch | {_package_version("torch")} ({device}) |
| Platform | {platform.platform()} |
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure what alpha query expansion and k-reciprocal "
        "re-ranking buy the image store on the Oxford Flower dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "benchmarks" / "reranking.md",
        help="Markdown file the report is written to.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=_REPO_ROOT
        / "docs"
        / "neural_networks"
        / "benchmarks"
        / "clip_retrieval_precision_recall.png",
        help="PNG file the precision-recall curves are written to.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=100,
        help="Number of results returned per query and scored.",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=200,
        help="Number of candidates retrieved per query before the re-ranking.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries sampled from each split, all by default.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size of the embedder."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the query sampling."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweeps, score the configurations and write the report."""
    args = _parse_args(argv)
    if args.pool < args.depth:
        raise ValueError("'--pool' must be at least '--depth'.")
    store, path_labels = _build_store(args.batch_size)
    class_sizes = Counter(path_labels.values())
    device = str(store.embedder.device)

    tuning = _load_split(_TUNING_SPLIT, args.num_queries, args.seed)
    rankings, seconds = _retrieve(store, tuning, args.depth, None)
    plain_tuning = _score(
        rankings, tuning, path_labels, class_sizes, args.depth, seconds
    )
    _log(
        f"[{tuning.name}] plain: mAP@{args.depth} {plain_tuning.mean_average_precision:.4f}"
    )
    expansion_scores = _sweep_expansion(
        store, tuning, path_labels, class_sizes, args.depth
    )
    best_expansion = _best(expansion_scores)
    pools, _ = _retrieve(store, tuning, args.pool, best_expansion)
    reranking_scores = _sweep_reranking(
        store, pools, tuning, path_labels, class_sizes, args.depth
    )
    best_reranking = _best(reranking_scores)
    num_tuning = len(tuning.images)
    del tuning, pools

    test = _load_split(_TEST_SPLIT, args.num_queries, args.seed)
    rankings, seconds = _retrieve(store, test, args.depth, None)
    plain = _score(rankings, test, path_labels, class_sizes, args.depth, seconds)
    rankings, seconds = _retrieve(store, test, args.depth, best_expansion)
    expanded = _score(rankings, test, path_labels, class_sizes, args.depth, seconds)
    pools, pool_seconds = _retrieve(store, test, args.pool, best_expansion)
    reranker = KReciprocalReranker(
        store,
        k1=best_reranking.k1,
        k2=best_reranking.k2,
        lambda_value=best_reranking.lambda_value,
    )
    rankings, rerank_seconds = _rerank(reranker, pools, args.depth)
    reranked = _score(
        rankings,
        test,
        path_labels,
        class_sizes,
        args.depth,
        pool_seconds + rerank_seconds,
    )
    results = [
        ("Plain retrieval", plain),
        (f"Alpha query expansion ({best_expansion.describe()})", expanded),
        (
            f"Alpha query expansion and k-reciprocal re-ranking "
            f"({best_reranking.describe()})",
            reranked,
        ),
    ]
    for label, scores in results:
        _log(
            f"[{test.name}] {label}: recall@1 {scores.recall_at_1:.4f}, "
            f"recall@5 {scores.recall_at_5:.4f}, mAP@{args.depth} "
            f"{scores.mean_average_precision:.4f}, MRR {scores.mean_reciprocal_rank:.4f}"
        )

    curves = {
        "Plain retrieval": plain.precision_at_recall,
        "Alpha query expansion": expanded.precision_at_recall,
        "Expansion and k-reciprocal re-ranking": reranked.precision_at_recall,
    }
    _plot_precision_recall(curves, len(test.images), args.depth, args.plot)
    _log(f"Wrote {args.plot}")
    report = _format_report(
        results,
        expansion_scores,
        best_expansion,
        reranking_scores,
        best_reranking,
        plain_tuning,
        len(store),
        num_tuning,
        len(test.images),
        args.depth,
        args.pool,
        device,
        args.plot,
        args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    _log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
