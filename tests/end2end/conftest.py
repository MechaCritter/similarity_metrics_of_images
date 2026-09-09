"""Fixtures shared by the end-to-end retrieval tests."""

from __future__ import annotations

import pytest

from pyvisim.datasets import OxfordFlowerDataset

from ._retrieval import Gallery, build_gallery

#: Split the gallery is built from.
GALLERY_SPLIT = "train"

#: Split every image of which is used as a query.
QUERY_SPLIT = "test"


@pytest.fixture(scope="session")
def gallery() -> Gallery:
    """The whole gallery split, embedded once for the whole session.

    :returns: the store over the gallery split and its labels.
    """
    return build_gallery(GALLERY_SPLIT)


@pytest.fixture(scope="session")
def queries() -> OxfordFlowerDataset:
    """The whole query split, left on disk until a chunk of it is scored.

    :returns: the dataset the query images are decoded from.
    """
    return OxfordFlowerDataset(purpose=QUERY_SPLIT)
