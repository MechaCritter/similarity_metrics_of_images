# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import importlib.util

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyvisim"
copyright = "2026, Nhat Huy Vu"
author = "Nhat Huy Vu"

release = importlib.metadata.version("pyvisim")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
# 'intro.rst' files are fragments pulled into the section index via
# '.. include::', so they must not be built as standalone documents.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/intro.rst"]

language = "en"

# The narrative pages included from docs/ link to repository source files
# (e.g. ../pyvisim/classic/vlad.py), which have no HTML equivalent.
# Their H1 titles are skipped via ':start-line: 1' so that each rST page owns
# the top-level heading, which makes the included content start at H2.
suppress_warnings = ["myst.xref_missing", "myst.header"]

# -- Autodoc -------------------------------------------------------------------

# The 'nn' extra is not required to build the documentation: any optional
# dependency that is missing from the environment is mocked so that autodoc can
# still import every module.
autodoc_mock_imports = [
    module
    for module in ("torch", "torchvision", "torchaudio", "open_clip")
    if importlib.util.find_spec(module) is None
]

autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- MyST (Markdown) -----------------------------------------------------------

# The Sphinx sources are reStructuredText; myst_parser is only needed to parse
# the narrative Markdown pages under docs/ that are pulled in via
# '.. include:: ... :parser: myst_parser.sphinx_'.
# dollarmath renders the $$...$$ formulas
myst_enable_extensions = ["colon_fence", "dollarmath"]
# Generate anchors for headings so links such as 'vlad.md#section' resolve.
myst_heading_anchors = 3

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"pyvisim {release}"
