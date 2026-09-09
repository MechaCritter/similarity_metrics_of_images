"""Tests for :class:`pyvisim.image_store.Candidate`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyvisim.image_store import Candidate


@pytest.fixture(scope="module")
def image_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, np.ndarray]:
    """Write a small RGB image to disk.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :returns: the file path and the ``(8, 6, 3)`` ``uint8`` array it holds.
    """
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8)
    path = tmp_path_factory.mktemp("candidate") / "match.png"
    Image.fromarray(array).save(path)
    return str(path), array


def test_exposes_path_and_score() -> None:
    """A candidate reports the path and the score it was built with."""
    candidate = Candidate("match.png", 0.25)
    assert candidate.path == "match.png"
    assert candidate.score == 0.25
    assert "match.png" in repr(candidate)


def test_score_is_stored_as_a_float() -> None:
    """Any real number is accepted as the score and kept as a float."""
    assert isinstance(Candidate("match.png", np.float32(0.5)).score, float)
    assert Candidate("match.png", 1).score == 1.0


def test_compares_by_path_and_score() -> None:
    """Two candidates are equal when their path and score are."""
    assert Candidate("match.png", 0.25) == Candidate("match.png", 0.25)
    assert Candidate("match.png", 0.25) != Candidate("other.png", 0.25)
    assert Candidate("match.png", 0.25) != Candidate("match.png", 0.5)
    assert len({Candidate("match.png", 0.25), Candidate("match.png", 0.25)}) == 1


def test_is_immutable() -> None:
    """The path and the score cannot be changed once the candidate is built."""
    candidate = Candidate("match.png", 0.25)
    with pytest.raises(AttributeError):
        candidate.score = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        candidate.path = "other.png"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "score"),
    [
        (Path("match.png"), 0.25),
        (123, 0.25),
        ("match.png", "0.25"),
        ("match.png", True),
    ],
)
def test_rejects_wrong_field_types(path: object, score: object) -> None:
    """The path must be a string and the score a number."""
    with pytest.raises(TypeError):
        Candidate(path, score)  # type: ignore[arg-type]


def test_array_reads_the_image_on_first_access(
    image_file: tuple[str, np.ndarray],
) -> None:
    """``array`` is the matched image, read from disk when first asked for."""
    path, expected = image_file
    candidate = Candidate(path, 0.0)
    assert "array" not in vars(candidate)
    array = candidate.array
    assert array.dtype == np.uint8
    assert np.array_equal(array, expected)


def test_array_is_read_once(image_file: tuple[str, np.ndarray]) -> None:
    """Later accesses hand back the array that was read the first time."""
    path, _ = image_file
    candidate = Candidate(path, 0.0)
    assert candidate.array is candidate.array


def test_array_of_a_missing_file_raises(tmp_path: Path) -> None:
    """A path that cannot be read is reported when the image is asked for."""
    candidate = Candidate(str(tmp_path / "gone.png"), 0.0)
    with pytest.raises(FileNotFoundError):
        _ = candidate.array


def test_array_does_not_take_part_in_equality(
    image_file: tuple[str, np.ndarray],
) -> None:
    """Reading the image changes nothing about what the candidate is."""
    path, _ = image_file
    read, unread = Candidate(path, 0.0), Candidate(path, 0.0)
    _ = read.array
    assert read == unread


def test_clear_buffer_evicts_the_read_image(
    image_file: tuple[str, np.ndarray],
) -> None:
    """``clear_buffer`` drops the kept array, and the next access reads it again."""
    path, expected = image_file
    candidate = Candidate(path, 0.0)
    first = candidate.array
    assert "array" in vars(candidate)
    candidate.clear_buffer()
    assert "array" not in vars(candidate)
    again = candidate.array
    assert again is not first
    assert np.array_equal(again, expected)


def test_clear_buffer_before_any_read_changes_nothing(
    image_file: tuple[str, np.ndarray],
) -> None:
    """Clearing a candidate whose image was never read is harmless."""
    path, _ = image_file
    candidate = Candidate(path, 0.0)
    candidate.clear_buffer()
    assert "array" not in vars(candidate)
    assert candidate == Candidate(path, 0.0)
