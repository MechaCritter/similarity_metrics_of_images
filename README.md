<!-- Logo -->
<p align="center">
  <img src="res/images/logo.png" alt="pyvisim" width="1418" />
</p>

<!-- Added badges to convey project readiness/branding (example placeholders) -->
![License](https://img.shields.io/github/license/MechaCritter/Python-Visual-Similarity)
![Version](https://img.shields.io/pypi/v/pyvisim)
![Status](https://img.shields.io/badge/status-pre--release-orange)
![Python](https://img.shields.io/pypi/pyversions/pyvisim)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

# Welcome to `pyvisim`!

`pyvisim` is a computer vision library for computing image similarities using traditional and deep learning methods.

📚 **Documentation**: <https://mechacritter.github.io/Python-Visual-Similarity/>

## Table of Contents

- [Status](#status)
- [Overview](#overview)
  - [Accelerated Computation](#accelerated-computation)
  - [Examples](#examples)
- [Installation](#installation)
- [Contributing](#contributing)
- [Get in Touch](#get-in-touch)
- [License](#license)

## Status

> [!WARNING]
> This project is still in early development, so the API might change anytime (with deprecation,
> but the change will come soon afterwards). Feel free to use it in development environments, but I
> would recommend against using it in production.
>
> The first stable release will have the version tag `v1.0.0` and will come approximately by the
> end of `August 2026`.

## Overview

![Architecture Diagram](https://raw.githubusercontent.com/MechaCritter/Python-Visual-Similarity/assets/docs/architecture/image_embeddings.drawio.png)

The goal of `pyvisim` is to become the largest collection of image similarity metrics, varying from
traditional methods like `PSNR`, `SSIM`, `Fisher Vectors`, and `VLAD` to deep learning methods like `CLIP` and `Siamese Networks`. Then, one can use these for image retrieval and clustering.

Currently, one would need to install numerous libraries just to get all the metrics mentioned (for example, `scikit-image` + `opencv-python` for `Fisher Vectors` and `SSIM`, `open-clip` for `CLIP Embedder`). `pyvisim`
attempts to close this gap by implementing as many metrics as possible using only `numpy`, `scipy` (for conventional metrics), and
`torch` (for deep learning metrics), plus making them more user-friendly with a simple Object-Oriented code design.

### Accelerated Computation

**Cython** kernels and **C++ libraries** are used for some metrics to accelerate computation significantly compared
to all reference libraries on the CPU. See, for example, [benchmark results of the `SSIM` implementation](docs/structural/README.md#benchmarking).

### Examples

#### `Structural Similarity` (see documentation [here](https://mechacritter.github.io/Python-Visual-Similarity/structural/index.html)):

```python
from pyvisim.structural import SSIM

ssim = SSIM()
similarity_score = ssim.similarity_score(image1, image2)
print(f"Similarity Score: {similarity_score}")
```

#### One-Shot similarity computation using the `CLIPEmbedder`(see documentation [here](https://mechacritter.github.io/Python-Visual-Similarity/neural_networks/clip.html)):

```python
from pyvisim.neural_networks import ClipEmbedder

# Declare the Clip Embedder
embedder = ClipEmbedder()

# Compute the similarity score. By default, cosine similarity is used.
similarity_score = embedder.similarity_score(image1, image2)
print(f"Similarity Score: {similarity_score}")
```

#### `Image retrieval` (see documentation [here](https://mechacritter.github.io/Python-Visual-Similarity/image_retrieval/image_store.html)):

```python
from pyvisim.neural_networks import ClipEmbedder
from pyvisim.image_store import InMemoryImageEmbeddingStore

embedder = ClipEmbedder()

image_store = InMemoryImageEmbeddingStore(
    image_paths=train_image_paths,
    embedder=embedder,
    search_index="hnsw",
    index_params={"graph_degree": 16, "build_candidates": 200},
)

candidates = image_store.retrieve_top_k_similar(image, k=5)[0] # returns a tuple of (image_path, similarity_score)
```

For more examples, please refer to the [`pyvisim` Examples
Repository](https://github.com/MechaCritter/Python-Visual-Similarity-Examples).

## Installation

To install the slim version (**without** deep learning features):

```bash
pip install pyvisim
```

Additional features include (note: these pull in heavy dependencies like `torch`):

```bash
# For deep learning features and the OxfordFlowerDataset
pip install "pyvisim[nn]"
```

All experiments in this project was made on the Oxford Flower Dataset
<ref>[7]</ref>, for which I have created a custom dataset class. For
more details on the dataset, please refer to the [documentation](pyvisim/datasets/README.md).

## Contributing

See [the contributing guidelines](CONTRIBUTING.md).

## Get in Touch
If you have any questions or just want to say hi, feel free to:
- Open an issue on [GitHub](https://github.com/MechaCritter/similarity_metrics_of_images/issues).
- Write me an email at [vunhathuy234@gmail.com](mailto:vunhathuy234@gmail.com).
- Connect on [LinkedIn](https://www.linkedin.com/in/nhat-huy-vu-80495111b/) to follow my work and share your thoughts.

## License
This project is licensed under the terms of the MIT license.
