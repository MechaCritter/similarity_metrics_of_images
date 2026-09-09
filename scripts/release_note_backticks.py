"""Normalize inline code markup in reno release notes.

Release notes are rendered as reStructuredText, where inline code is wrapped in
double backticks. Markdown habits leak in as single backticks, which render as
plain italics-like text, and as triple-backtick fences, which do not render at
all. The 'Check Release Notes' workflow rejects both, so this hook rewrites
what it can before the CI ever sees it.

Run as a pre-commit hook over 'releasenotes/notes/*.yaml', or by hand::

    python scripts/release_note_backticks.py --check releasenotes/notes/*.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: A single-backtick span that is not part of a double or triple backtick run.
#: The lookbehind for ':' spares reStructuredText roles such as ':class:`X`'
#: and the lookahead for '_' spares hyperlink references such as '`text`_'.
SINGLE_BACKTICK_RE = re.compile(r"(?<![`:])`(?!`)([^`\n]+?)`(?![`_])")

TRIPLE_BACKTICK = "```"

CODE_BLOCK_HINT = (
    "Found a triple backtick fence. Use the reStructuredText directive "
    "'.. code:: python' instead."
)


def rewrite(text: str) -> str:
    """Wrap every single-backtick span of ``text`` in double backticks.

    :param text: The contents of a release note.
    :return: The contents with single backticks replaced by double backticks.
    """
    return SINGLE_BACKTICK_RE.sub(r"``\1``", text)


def find_fences(text: str) -> list[int]:
    """Collect the line numbers of ``text`` that hold a triple backtick fence.

    :param text: The contents of a release note.
    :return: The 1-based line numbers carrying a fence.
    """
    return [
        line_no
        for line_no, line in enumerate(text.splitlines(), start=1)
        if TRIPLE_BACKTICK in line
    ]


def process(path: Path, check_only: bool) -> list[str]:
    """Check one release note, rewriting it unless ``check_only`` is set.

    :param path: The release note to process.
    :param check_only: Report problems without touching the file.
    :return: The problems found, as printable messages. Empty when the note is
        already clean, or when it was repaired in place.
    """
    text = path.read_text(encoding="utf-8")
    problems = [f"{path}:{line_no}: {CODE_BLOCK_HINT}" for line_no in find_fences(text)]

    rewritten = rewrite(text)
    if rewritten == text:
        return problems

    if check_only:
        problems.append(f"{path}: Found single backticks. Use double backticks.")
        return problems

    path.write_text(rewritten, encoding="utf-8")
    problems.append(f"{path}: Rewrote single backticks to double backticks.")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Process every release note named on the command line.

    :param argv: The command line arguments, defaulting to ``sys.argv[1:]``.
    :return: ``0`` when every note was already clean, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Release notes to process")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report problems without rewriting any file",
    )
    args = parser.parse_args(argv)

    problems = [problem for path in args.files for problem in process(path, args.check)]
    for problem in problems:
        print(problem, file=sys.stderr)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
