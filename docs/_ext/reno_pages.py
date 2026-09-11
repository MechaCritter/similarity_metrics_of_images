"""Generate one documentation page per release from the reno release notes.

The furo sidebar lists documents, not the sections inside them. For every
release to appear as its own entry under the release notes page, each version
reported by reno is written to a page of its own when the build starts,
together with the toctree that the release notes page includes.
"""

import logging
from pathlib import Path
from typing import Any

from reno import config, formatter, loader
from sphinx.application import Sphinx

# Relative to the Sphinx source directory. The release notes page sits in the
# parent folder and includes TOCTREE_FILE from here.
PAGES_DIR = Path("release_notes") / "versions"
TOCTREE_FILE = "toctree.inc"

UNRELEASED_VERSION_TITLE = "Unreleased"


def render_versions(repo_root: Path) -> dict[str, str]:
    """Render the release notes of every version reported by reno.

    Notes merged since the last version tag belong to a version named after
    that tag and the number of commits since, such as ``v0.9.4-3``. Its page is
    titled :data:`UNRELEASED_VERSION_TITLE`.

    :param repo_root: Root of the git repository holding ``releasenotes/``.
    :return: The reStructuredText page of each version, newest version first.
    """
    conf = config.Config(str(repo_root))
    conf.override(unreleased_version_title=UNRELEASED_VERSION_TITLE)
    with loader.Loader(conf, ignore_cache=True) as ldr:
        return {
            version: formatter.format_report(ldr, conf, [version], show_source=False)
            for version in ldr.versions
        }


def format_toctree(page_names: list[str]) -> str:
    """Return a toctree that lists the given version pages in order.

    :param page_names: Names of the version pages, without the file suffix.
    :return: The toctree directive, to be included by the release notes page.
    """
    entries = "".join(f"   {PAGES_DIR.name}/{name}\n" for name in page_names)
    return f".. toctree::\n   :maxdepth: 1\n\n{entries}"


def write_if_changed(path: Path, text: str) -> None:
    """Write the text to the file unless the file already holds it.

    An unchanged page keeps its modification time, so an incremental build
    does not read it again.

    :param path: File to write.
    :param text: Content of the file.
    """
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def remove_stale_pages(pages_dir: Path, page_names: list[str]) -> None:
    """Delete the version pages that are not listed anymore.

    This covers the page of the unreleased version once it has been tagged,
    which the toctree no longer references.

    :param pages_dir: Folder holding the version pages.
    :param page_names: Names of the pages to keep, without the file suffix.
    """
    for page in pages_dir.glob("*.rst"):
        if page.stem not in page_names:
            page.unlink()


def generate_pages(app: Sphinx) -> None:
    """Write the page of every version and the toctree that lists them.

    :param app: The Sphinx application, whose source directory lives in the
        root of the git repository.
    """
    srcdir = Path(app.srcdir)
    pages_dir = srcdir / PAGES_DIR
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = render_versions(srcdir.parent)
    for name, text in pages.items():
        write_if_changed(pages_dir / f"{name}.rst", text)
    remove_stale_pages(pages_dir, list(pages))
    write_if_changed(pages_dir / TOCTREE_FILE, format_toctree(list(pages)))


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the page generation with Sphinx.

    :param app: The Sphinx application.
    :return: The extension metadata.
    """
    # furo's 'a11y_pygments' dependency sets the root logger to INFO on import,
    # which would print every commit that reno scans into the build output.
    logging.getLogger("reno").setLevel(logging.WARNING)
    app.connect("builder-inited", generate_pages)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
