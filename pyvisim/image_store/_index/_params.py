"""
Vendor-neutral index parameters and their translation to a backend's own.

The indexes take parameters named after what they do rather than after the
library that implements them. Each index owns one table mapping those names
onto the keywords its backend actually understands, so a caller's vocabulary
stays put when the backend behind an index changes: only the table moves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: HNSW graph parameters, keyed by the name this library exposes and mapped
#: onto the hnswlib keyword each one drives.
HNSW_TO_HNSWLIB: dict[str, str] = {
    "space": "space",
    "capacity": "max_elements",
    "graph_degree": "M",
    "build_candidates": "ef_construction",
    "search_candidates": "ef",
    "random_seed": "random_seed",
    "num_threads": "num_threads",
}

#: Brute-force parameters, mapped the same way.
BRUTE_FORCE_TO_HNSWLIB: dict[str, str] = {
    "space": "space",
    "capacity": "max_elements",
    "num_threads": "num_threads",
}

#: Parameters a store fills in itself and therefore does not accept in
#: ``index_params``: the metric space is a store-level argument of its own, and
#: the capacity follows from the gallery being indexed.
_STORE_SUPPLIED = frozenset({"space", "capacity"})


def accepted_index_params(mapping: Mapping[str, str]) -> tuple[str, ...]:
    """
    List the parameters a caller may pass to an index the store builds.

    :param mapping: The index's parameter table, such as
        :data:`HNSW_TO_HNSWLIB`.
    :return: The table's names minus the ones the store supplies itself.
    """
    return tuple(name for name in mapping if name not in _STORE_SUPPLIED)


def as_backend_params(
    params: Mapping[str, Any],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    """
    Rename the parameters onto the keywords the backend takes.

    :param params: Parameter values keyed by their vendor-neutral name.
    :param mapping: The index's parameter table, such as
        :data:`HNSW_TO_HNSWLIB`.
    :return: The same values, keyed by the backend's own names.
    :raises KeyError: If a name is missing from the table.
    """
    return {mapping[name]: value for name, value in params.items()}


def validate_index_params(
    params: Mapping[str, Any],
    mapping: Mapping[str, str],
    index_name: str,
) -> None:
    """
    Reject parameters the named index does not take.

    :param params: The parameters a caller asked the index to be built with.
    :param mapping: The index's parameter table, such as
        :data:`HNSW_TO_HNSWLIB`.
    :param index_name: Name of the index, used in the error message.
    :raises ValueError: If ``params`` holds a name the index does not accept.
    """
    accepted = accepted_index_params(mapping)
    unknown = sorted(name for name in params if name not in accepted)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} for the {index_name!r} index. "
            f"Accepted parameters are: {sorted(accepted)}."
        )
