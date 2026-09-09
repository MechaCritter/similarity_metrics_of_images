import argparse
import itertools
import os
import time
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure as tm_msssim,
)

from pyvisim.base import DenseMetricBase
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.structural import MSSSIM, SSIM
from pyvisim.typing import UInt8NumpyArray

_SCRIPT_REF = "docs/structural/benchmarks/generate_benchmark.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Fixed thread budget of every run.
_NUM_THREADS = 4

_DATASET_SPLIT = "train"
_NOISE_STD = 15.0

_IMAGE_BASE_URL = "https://mechacritter.github.io/Python-Visual-Similarity/benchmarks"

# Chart chrome (light mode) and the two fixed series slots, following the
# validated default dataviz palette: blue = pyvisim, green = baseline.
_COLOR_PYVISIM = "#2a78d6"
_COLOR_BASELINE = "#008300"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"

#: One sampled image: (base file name, RGB uint8 array).
_NamedImage = tuple[str, UInt8NumpyArray]
_Gallery = list[UInt8NumpyArray]


def _package_version(name: str) -> str:
    """Resolve an installed package version, or ``"unknown"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _sample_disjoint_images(
    dataset: OxfordFlowerDataset, rng: np.random.Generator, counts: Sequence[int]
) -> list[list[_NamedImage]]:
    """Sample disjoint groups of named dataset images, one group per count."""
    total = sum(counts)
    if total > len(dataset):
        raise ValueError(
            f"Cannot sample {total} distinct images from a dataset of "
            f"{len(dataset)} images."
        )

    def named(index: int) -> _NamedImage:
        image, _, path = dataset[index]
        return os.path.basename(path), image

    indices = iter(rng.choice(len(dataset), size=total, replace=False))
    return [
        [named(int(i)) for i in itertools.islice(indices, count)] for count in counts
    ]


def _resize(image: UInt8NumpyArray, size: int) -> UInt8NumpyArray:
    """Resize an RGB uint8 image to a square side length."""
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    return cast(UInt8NumpyArray, np.asarray(resized))


def _add_noise(
    image: UInt8NumpyArray, std: float, rng: np.random.Generator
) -> UInt8NumpyArray:
    """Add clipped Gaussian noise to a uint8 image."""
    noisy = image.astype(np.float64) + rng.normal(0.0, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _skimage_ssim(image_a: UInt8NumpyArray, image_b: UInt8NumpyArray) -> float:
    """Score one RGB pair with the scikit-image SSIM baseline (Wang et al. 2004)."""
    return float(
        structural_similarity(
            image_a,
            image_b,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=255,
            channel_axis=-1,
        )
    )


def _to_bchw(images: Iterable[UInt8NumpyArray], dtype: type) -> torch.Tensor:
    """Stack same-shape RGB images into a ``(B, C, H, W)`` torch tensor."""
    stacked = np.stack(list(images)).astype(dtype).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def _torchmetrics_msssim(rows: torch.Tensor, cols: torch.Tensor) -> float:
    """Score aligned ``(B, C, H, W)`` pairs with the torchmetrics MS-SSIM baseline."""
    return float(
        tm_msssim(
            rows, cols, data_range=255.0, kernel_size=11, sigma=1.5, normalize="relu"
        )
    )


def _measure_accuracy(
    metric: DenseMetricBase,
    baseline_fn: Callable[[UInt8NumpyArray, UInt8NumpyArray], float],
    images: Sequence[_NamedImage],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Score each image against a noise-distorted copy of itself, per file name."""
    results: dict[str, dict[str, float]] = {}
    for name, image in images:
        distorted = _add_noise(image, _NOISE_STD, rng)
        ours = float(metric.similarity_score(image, distorted)[0, 0])
        ref = baseline_fn(image, distorted)
        results[name] = {"pyvisim": ours, "baseline": ref, "abs_error": abs(ours - ref)}
    return results


def _time_ms(fn: Callable[[], object], repeats: int) -> list[float]:
    """Measure repeated calls in milliseconds, after one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(1000.0 * (time.perf_counter() - start))
    return times


def _time_skimage_matrix(
    gallery_a: _Gallery, gallery_b: _Gallery, repeats: int
) -> list[float]:
    """Time the scikit-image baseline over all pairs, via a Python loop."""
    return _time_ms(
        lambda: [_skimage_ssim(a, b) for a in gallery_a for b in gallery_b], repeats
    )


def _time_torchmetrics_matrix(
    gallery_a: _Gallery, gallery_b: _Gallery, repeats: int
) -> list[float]:
    """
    Time the torchmetrics baseline over all pairs, as one stacked batch.

    The tensor stacking is done once outside the timed region, mirroring the
    benchmark notebook (pyvisim timings, by contrast, always include the full
    input pipeline).
    """
    stack_a = _to_bchw(gallery_a, np.float32)
    stack_b = _to_bchw(gallery_b, np.float32)
    rows = stack_a.repeat_interleave(stack_b.shape[0], dim=0)
    cols = stack_b.repeat(stack_a.shape[0], 1, 1, 1)
    return _time_ms(lambda: _torchmetrics_msssim(rows, cols), repeats)


def _measure_runtime(
    metric: DenseMetricBase,
    baseline_timer: Callable[[_Gallery, _Gallery, int], list[float]],
    groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> dict[str, dict[str, object]]:
    """
    Time every runtime workload for one metric and its baseline.

    ``groups`` holds the sampled images for, in order: the single pair, the
    expanded pair, the batch of 4 and the batch of 8. Every image of a
    scenario's first gallery is scored against every image of its second.
    """
    single_group, expanded_group, small_group, large_group = groups
    single = [_resize(image, 512) for _, image in single_group]
    expanded = _resize(expanded_group[0][1], 1024)
    distorted = _add_noise(expanded, _NOISE_STD, rng)
    small = [_resize(image, 256) for _, image in small_group]
    large = [_resize(image, 256) for _, image in large_group]
    scenarios: list[tuple[str, _Gallery, _Gallery, list[_NamedImage], int]] = [
        ("1 pair (512x512)", [single[0]], [single[1]], single_group, 512),
        ("1 expanded pair (1024x1024)", [expanded], [distorted], expanded_group, 1024),
        ("batch of 4 (16 pairs)", small, small, small_group, 256),
        ("batch of 8 (64 pairs)", large, large, large_group, 256),
    ]
    results: dict[str, dict[str, object]] = {}
    for label, gallery_a, gallery_b, group, size in scenarios:
        pyvisim_ms = _time_ms(
            lambda a=gallery_a, b=gallery_b: metric.similarity_score(a, b), repeats
        )
        results[label] = {
            "image_names": [name for name, _ in group],
            "image_size": size,
            "n_pairs": len(gallery_a) * len(gallery_b),
            "pyvisim_ms": pyvisim_ms,
            "baseline_ms": baseline_timer(gallery_a, gallery_b, repeats),
        }
    return results


def _format_ms(value: float) -> str:
    """Format a duration in milliseconds for labels and tables."""
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def _plot_runtime_barplot(
    runtime: dict[str, dict[str, object]], baseline_label: str, title: str, path: Path
) -> None:
    """Plot the median pyvisim vs. baseline runtime per scenario as grouped bars."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pos = np.arange(len(runtime))
    medians = {
        "pyvisim": [float(np.median(r["pyvisim_ms"])) for r in runtime.values()],
        baseline_label: [float(np.median(r["baseline_ms"])) for r in runtime.values()],
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    width = 0.38
    for (label, values), color, offset in zip(
        medians.items(), (_COLOR_PYVISIM, _COLOR_BASELINE), (-0.5, 0.5), strict=True
    ):
        x = pos + offset * width
        bars = ax.bar(x, values, width=width, color=color, label=label)
        texts = [_format_ms(value) for value in values]
        ax.bar_label(bars, labels=texts, color=_INK_SECONDARY, fontsize=8, padding=2)
    ax.set_xticks(pos)
    # Break each label before its parenthesis into a two-line tick.
    ax.set_xticklabels([label.replace(" (", "\n(") for label in runtime])
    ax.set_ylabel("median runtime per call (ms)", color=_INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=_INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _format_accuracy_table(accuracy: dict[str, dict[str, float]]) -> str:
    """Render the per-image accuracy results as a markdown table."""
    lines = ["| Image | pyvisim | Baseline | Abs. error |", "|---|---|---|---|"]
    for name in sorted(accuracy):
        s = accuracy[name]
        lines.append(
            f"| `{name}` | {s['pyvisim']:.6f} | {s['baseline']:.6f} "
            f"| {s['abs_error']:.2e} |"
        )
    return "\n".join(lines)


def _format_runtime_table(runtime: dict[str, dict[str, object]]) -> str:
    """Render the per-scenario median runtimes as a markdown table."""
    lines = [
        "| Scenario | Images | pyvisim (ms) | Baseline (ms) | Speed-up |",
        "|---|---|---|---|---|",
    ]
    for label, result in runtime.items():
        ours = float(np.median(result["pyvisim_ms"]))
        ref = float(np.median(result["baseline_ms"]))
        names = ", ".join(f"`{name}`" for name in result["image_names"])
        size = result["image_size"]
        lines.append(
            f"| {label}, {size}x{size} | {names} "
            f"| {_format_ms(ours)} | {_format_ms(ref)} | {ref / ours:.1f}x |"
        )
    return "\n".join(lines)


def _format_document(bench: dict[str, object], seed: int, repeats: int) -> str:
    """Render the complete generated ``benchmark.md`` of one metric."""
    accuracy = cast(dict[str, dict[str, float]], bench["accuracy"])
    runtime = cast(dict[str, dict[str, object]], bench["runtime"])
    return f"""## Benchmark

> [!IMPORTANT]
> This file was generated by the script
> `{_SCRIPT_REF}`. **Do not edit manually!**

All images are drawn from the Oxford Flower dataset
(`{_DATASET_SPLIT}` split, seed {seed}), with a disjoint image subset per
experiment and per metric. `num_workers={_NUM_THREADS}`. {repeats} timed
calls after one warm-up.

Baseline: **{bench["baseline"]}**.

### Accuracy

Each image (native resolution) is scored against a copy distorted with
Gaussian noise (std {_NOISE_STD:g}). The error is the absolute difference
between the pyvisim score and the baseline score.

{_format_accuracy_table(accuracy)}

### Runtime

Median wall-clock time per full scoring call ({bench["metric_label"]} of every
image in the first gallery against every image in the second). For a fair comparison,
GPU is disabled.

{_format_runtime_table(runtime)}

![{bench["metric_label"]} median runtime]({_IMAGE_BASE_URL}/{bench["slug"]}_runtime_barplot.png)
"""


def _write_document(docs_dir: Path, slug: str, document: str) -> Path:
    """
    Write the generated ``benchmark.md`` of one metric, replacing any
    previous one.

    :param docs_dir: Documentation folder of the module.
    :param slug: Folder name of the metric inside ``docs_dir``.
    :param document: Rendered Markdown of the metric.
    :return: The path the document was written to.
    """
    path = docs_dir / slug / "benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
    return path


def _benchmark_ssim(
    accuracy_images: list[_NamedImage],
    runtime_groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> dict[str, object]:
    """Benchmark :class:`SSIM` against scikit-image."""
    metric = SSIM(num_workers=_NUM_THREADS)
    return {
        "metric_label": "SSIM",
        "slug": "ssim",
        "baseline_label": "scikit-image",
        "baseline": (
            f"scikit-image {_package_version('scikit-image')} — "
            "skimage.metrics.structural_similarity(win_size=11, "
            "gaussian_weights=True, sigma=1.5, use_sample_covariance=False, "
            "data_range=255)"
        ),
        "accuracy": _measure_accuracy(metric, _skimage_ssim, accuracy_images, rng),
        "runtime": _measure_runtime(
            metric, _time_skimage_matrix, runtime_groups, rng, repeats
        ),
    }


def _benchmark_msssim(
    accuracy_images: list[_NamedImage],
    runtime_groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> dict[str, object]:
    """Benchmark :class:`MSSSIM` against torchmetrics."""

    def baseline_pair(image_a: UInt8NumpyArray, image_b: UInt8NumpyArray) -> float:
        return _torchmetrics_msssim(
            _to_bchw([image_a], np.float64), _to_bchw([image_b], np.float64)
        )

    metric = MSSSIM(num_workers=_NUM_THREADS)
    return {
        "metric_label": "MS-SSIM",
        "slug": "ms_ssim",
        "baseline_label": "torchmetrics",
        "baseline": (
            f"torchmetrics {_package_version('torchmetrics')} — "
            "torchmetrics.functional.image."
            "multiscale_structural_similarity_index_measure(data_range=255.0, "
            "kernel_size=11, sigma=1.5, normalize='relu')"
        ),
        "accuracy": _measure_accuracy(metric, baseline_pair, accuracy_images, rng),
        "runtime": _measure_runtime(
            metric, _time_torchmetrics_matrix, runtime_groups, rng, repeats
        ),
    }


def _write_outputs(
    benchmarks: Sequence[dict[str, object]],
    output_dir: Path,
    docs_dir: Path,
    seed: int,
    repeats: int,
) -> None:
    """Render the runtime plots and rewrite the generated benchmark pages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for bench in benchmarks:
        image_path = output_dir / f"{bench['slug']}_runtime_barplot.png"
        _plot_runtime_barplot(
            cast(dict[str, dict[str, object]], bench["runtime"]),
            cast(str, bench["baseline_label"]),
            f"{bench['metric_label']}: median runtime, pyvisim vs. "
            f"{bench['baseline_label']}",
            image_path,
        )
        print(f"Wrote {image_path}")
    for bench in benchmarks:
        document_path = _write_document(
            docs_dir,
            cast(str, bench["slug"]),
            _format_document(bench, seed, repeats),
        )
        print(f"Wrote {document_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate the "Benchmark" documentation pages '
        "for ``pyvisim.structural``."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "structural" / "benchmarks",
        help="Directory the runtime barplot PNGs are written to.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "structural",
        help="Documentation folder holding one sub-folder per metric.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the image sampling."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of timed calls per runtime workload.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run every benchmark and write the outputs."""
    args = _parse_args(argv)
    os.environ["PYVISIM_NUM_THREADS"] = str(_NUM_THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    if torch.cuda.is_available():
        raise RuntimeError("GPU is available. This comparison is not fair.")
    torch.set_num_threads(_NUM_THREADS)
    if int(torch.get_num_threads()) != int(os.environ["PYVISIM_NUM_THREADS"]):
        raise RuntimeError(
            f"torch.get_num_threads()={torch.get_num_threads()} "
            f"!= PYVISIM_NUM_THREADS={os.environ['PYVISIM_NUM_THREADS']}"
        )
    rng = np.random.default_rng(args.seed)
    dataset = OxfordFlowerDataset(purpose=_DATASET_SPLIT)
    # Per metric: 8 accuracy images plus the four disjoint runtime groups
    # (single pair, expanded pair, batch of 4, batch of 8).
    counts_per_metric = [8, 2, 1, 4, 8]
    groups = _sample_disjoint_images(dataset, rng, counts_per_metric * 2)
    benchmarks = (
        _benchmark_ssim(groups[0], groups[1:5], rng, args.repeats),
        _benchmark_msssim(groups[5], groups[6:10], rng, args.repeats),
    )
    _write_outputs(benchmarks, args.output_dir, args.docs_dir, args.seed, args.repeats)


if __name__ == "__main__":
    main()
