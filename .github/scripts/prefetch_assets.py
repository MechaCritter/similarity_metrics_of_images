"""
Download the CI assets that live outside the repository.

The Oxford Flowers dataset and the ImageNet backbone weights are fetched from
third-party servers the first time a test needs them. Both are large and change
very rarely, so CI restores them from the GitHub Actions cache instead. This
script populates that cache: it downloads whatever is missing and is a no-op
once the cache has been restored.

Run it with the ``nn`` extra installed::

    uv run python .github/scripts/prefetch_assets.py

``--print-cache-dir`` writes the directory holding the downloaded dataset to
stdout, so a workflow can point ``actions/cache`` at it without hard-coding one
path per runner OS::

    uv run python .github/scripts/prefetch_assets.py --print-cache-dir
"""

from __future__ import annotations

import argparse
import sys

from platformdirs import user_cache_dir

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks.backbones import build_backbone, list_backbones


def cache_dir() -> str:
    return user_cache_dir("pyvisim")


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def prefetch_dataset() -> None:
    report(f"Prefetching the Oxford Flowers dataset into {cache_dir()}")
    dataset = OxfordFlowerDataset(purpose="train")
    report(f"Oxford Flowers ready ({len(dataset)} training images)")


def prefetch_backbones() -> None:
    for backbone in list_backbones():
        report(f"Prefetching the {backbone} ImageNet weights")
        build_backbone(backbone, pretrained=True)
    report("Backbone weights ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch the CI assets.")
    parser.add_argument(
        "--print-cache-dir",
        action="store_true",
        help="print the dataset cache directory and exit without downloading",
    )
    args = parser.parse_args()

    if args.print_cache_dir:
        sys.stdout.write(cache_dir())
        return

    prefetch_dataset()
    prefetch_backbones()


if __name__ == "__main__":
    main()
