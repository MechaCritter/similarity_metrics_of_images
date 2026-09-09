"""The result type shared by the image store and the re-ranking of its results."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import cached_property

from .._utils import read_image_rgb
from ..typing import UInt8NumpyArray


@dataclass(frozen=True)
class Candidate:
    """
    A single retrieval result.

    The matched image itself is not read until :attr:`array` is first accessed.
    Hence, a ranked list of candidates only costs its paths and scores
    until the image array is actually needed.

    :param path: Path of the matched gallery image.
    :param score: Score the match was ranked by, best first. For the built-in
        indexes it is a distance, so lower means more similar.
    :raises TypeError: If ``path`` is not a string or ``score`` is not a
        number.
    """

    path: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"'path' must be a string, got {type(self.path).__name__}.")
        if isinstance(self.score, bool) or not isinstance(self.score, numbers.Real):
            raise TypeError(
                f"'score' must be a number, got {type(self.score).__name__}."
            )
        # The dataclass is frozen, so the coercion goes through the base setter.
        object.__setattr__(self, "score", float(self.score))

    @cached_property
    def array(self) -> UInt8NumpyArray:
        """
        The matched image as an RGB ``uint8`` array of shape ``(H, W, 3)``.

        The file is read on first access and the array is kept, so later
        accesses cost nothing.

        :raises FileNotFoundError: If the image cannot be read from ``path``.
        """
        return read_image_rgb(self.path)

    def clear_buffer(self) -> None:
        """Evict the cached :attr:`array` from memory, if it has been loaded."""
        self.__dict__.pop("array", None)
