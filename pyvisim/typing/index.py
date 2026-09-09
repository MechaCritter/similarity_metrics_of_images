"""
Structural type describing the search-index interface used by an image store.
"""

from collections.abc import Sequence
from typing import Protocol

from .numeric import Float32NumpyArray, FloatNumpyArray, IntNumpyArray


class SearchIndex(Protocol):
    """Protocol for indexes that accelerate nearest-neighbour search."""

    @property
    def vectors(self) -> Float32NumpyArray: ...

    @property
    def dim(self) -> int: ...

    def vectors_at(self, ids: Sequence[int] | IntNumpyArray) -> Float32NumpyArray:
        """
        Read the vectors stored under the given row numbers.

        :param ids: Gallery row numbers, shape ``(n,)``, at least one.
        :return: The ``(n, D)`` block of the requested vectors, in the given
            order.
        """
        ...

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]: ...
