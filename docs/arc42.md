# arc42: pyvisim

Software architecture of the library as a whole. This document is for
developers and is not part of the published documentation. Each module has its
own `arc42.md` next to its pages, and this one carries what spans them.

## Building block view

- [Typing](typing/arc42.md): the public types and the protocols the components
  are written against.
- [Distance](distance/arc42.md): the distance metrics that compare embeddings.
- [Structural](structural/arc42.md): SSIM and MSSSIM.
- [Pixelwise](pixelwise/arc42.md): PSNR.
- [Classic](classic/arc42.md): classical embedding methods pre deep learning
  era.
- [Image similarity retrieval](image_similarity_retrieval/arc42.md): image
  store, its search indexes and the re-ranking of its results.
- [Features](features/arc42.md): image feature extractors.
- [Neural networks](neural_networks/arc42.md): Siamese networks, triplet
  networks, CLIP embedders.
- [Dataset](dataset/arc42.md): `torch` datasets.
- [Eval](eval/arc42.md): retrieval scoring functions.

The abstract bases every public class derives from live in two places:
`pyvisim/_base_classes.py` (`SimilarityMetric`, `FeatureExtractorBase`,
`ImageEmbedderBase`, `SerializableImageEmbedder`) and `pyvisim/base/`
(`DenseMetricBase`, shared by the dense metrics).

## Architecture decisions

### Serialization uses the safetensors `.embedder` format

Pickling is explicitly avoided out of safety reasons: it mitigates the risk of
deserializing malicious objects. Arrays are written as
[safetensors](https://github.com/huggingface/safetensors), and the structure
plus the scalars travel as one JSON blob in the file metadata, with a class-name
registry dispatching a file back onto the class that wrote it.

`torch.save` and `torch.load` still work on the neural networks, as
conventionally used in PyTorch.

### Heavyweight dependencies are optional and imported lazily

`pyvisim` advertises heavyweight extras without forcing every user to install
them by introducing [Optional
Imports](https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/pyvisim/lazy_import).
An optional import is attempted eagerly, and if the dependency is missing, the
resulting `ImportError` is captured and only re-raised when the dependent code
is actually used.

The classical pipeline still *accepts* torch tensors when torch happens to be
installed, but it must not depend on torch. `is_tensor` captures that contract:
it short-circuits to `False` when torch is absent instead of raising.

### Vendored third-party code stays byte-identical to its source

Files under a `_vendored` folder are 1-to-1 copies of their original sources and stay
unchanged for the rest of their lifetime inside `pyvisim` so that, in case of
changes coming from the upstream, one would only need some text diff tool to
compare the changes, and simply overwrite the current files with the files from
the upstream. Where behavior changes are necessary, the developer adds a subclass or an
overriding method in a separate file.

## Risks and technical debt

- `pyvisim/_base_classes.py` should move into `pyvisim/base/`, so that all
  abstract bases live in one module.
- Add **tensor sketch approximation** and **mutual information** analysis for
  Fisher Vector, according to the paper by Weixia Zhang, Jia Yan, Wenxuan Shi,
  Tianpeng Feng, and Dexiang Deng.
- Add support for **vision transformers** for the `DeepConvFeature` class.
