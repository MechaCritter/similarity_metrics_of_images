"""The result type shared by the image store and the re-ranking of its results."""

from __future__ import annotations

from typing import NamedTuple


class Candidate(NamedTuple):
    """A single retrieval result.

    :param path: Path of the matched gallery image.
    :param score: Score the index ranked the match by, best first. For the
        built-in indexes it is a distance, so lower means more similar.
    """

    path: str
    score: float
