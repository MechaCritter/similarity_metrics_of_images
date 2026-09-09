.PHONY: test-types test-unit test-slow build-ext fmt docs release-note release-notes

# Regenerate the checked-in Cython C sources and rebuild the editable install.
# --inexact keeps ad-hoc packages in the venv from being pruned.
build-ext:
	uv run --group build cythonize -3 pyvisim/structural/_kernel/_ssim_kernels.pyx
	uv run --group build cythonize -3 pyvisim/features/_vendored/sift/_sift.pyx
	uv run --group build cythonize -3 pyvisim/pixelwise/_kernel/_ssd_kernel.pyx
	uv sync --inexact --reinstall-package pyvisim

# Strict mypy type-checking
test-types:
	uv run --group types --extra nn mypy pyvisim/

# Unit tests with a terminal coverage report (skips slow, weight-downloading tests)
test-unit:
	uv run --group test --extra nn pytest -m "not slow"

# Test slow tests
test-slow:
	uv run --group test --extra nn pytest -m slow
# Formatting with ruff
fmt:
	uv run --group fmt ruff check --fix .
	uv run --group fmt ruff format .

# Build the Sphinx HTML documentation for local review (same flags as CI);
# open docs/sphinx/_build/html/index.html afterwards
docs:
	uv run --group docs --extra nn sphinx-build -W -b html docs/sphinx docs/sphinx/_build/html

# Create a release note under releasenotes/notes/ for the current change.
# Usage: make release-note NAME=my-change
release-note:
	@test -n "$(NAME)" || echo "Usage: make release-note NAME=my-change" >&2
	@test -n "$(NAME)"
	uv run --group release reno new $(NAME)

# Render the accumulated release notes for local review
release-notes:
	uv run --group release reno report --no-show-source --ignore-cache
