# Changelog

> [!NOTE]
> This file is frozen. It documents releases up to and including `0.9.3`.
> Release notes for later versions are managed with
> [reno](https://docs.openstack.org/reno/latest/) and published on the
> [GitHub Releases page](https://github.com/MechaCritter/Python-Visual-Similarity/releases).
> See [CONTRIBUTING.md](CONTRIBUTING.md#release-notes) for more
information.

## [0.9.3] - 2026-09-09

### Added
- Every similarity metric now takes a `batch_size` argument, exposes it as the
  `batch_size` attribute and takes a new value through `set_batch_size`. It
  defaults to `16` everywhere; `-1` turns the splitting off and processes the
  whole input as one batch, whatever its size.
- Serializable embedders store their batch size, so a reloaded embedder runs
  with the batch size it was saved with.
- `VLADEmbedder` and `FisherVectorEmbedder` take a `batch_size`. The images of
  a batch are extracted, reduced, assigned and normalized as one matrix instead
  of one image at a time, and an iterable input stays a stream.
- `FeatureExtractorBase.extract_batch` extracts a batch of images and returns
  one feature array per image. Extractors that can do a whole batch in one go
  override it; the default extracts one image at a time.
- `DeepConvFeature` pushes a whole batch through its backbone in one forward
  pass. A custom `transform` that keeps the input size leaves the images
  unstackable, and they are then extracted one at a time as before.
- `ClipEmbedder`, `ContrastiveSiameseNetwork`, `TripletNeuralNetwork` and
  `BCESiameseNetwork` take a `batch_size` that splits their forward passes and
  bounds the activation memory of each.
- `Pipeline` takes a `batch_size` bounding how many images it hands its
  embedders at a time. Each embedder still applies its own batch size within.
- `InMemoryImageEmbeddingStore` embeds its gallery one batch at a time, sized by
  the `batch_size` of the embedder it is given, instead of one image per call.
- `InMemoryImageEmbeddingStore` takes a `num_workers` (default `4`) reading and
  decoding the gallery files on worker threads while the embedder works on the
  previous batch.
- `InMemoryImageEmbeddingStore` takes a `num_prefetch_batches` (default `4`)
  controlling how many batches of images the reading threads may run ahead of
  the embedder.
- `InMemoryImageEmbeddingStore.retrieve_top_k_similar` takes `query_expansion`,
  `expansion_alpha` and `expansion_neighbours` to refine every query with the
  alpha query expansion of Radenović et al. (2019) before the final search. It
  is off by default, since it costs one extra search per query.
- `KReciprocalReranker` (in `pyvisim.image_store`): re-ranks the candidates of a
  query with the k-reciprocal encoding of Zhong et al. (2017) through
  `rerank(candidates, top_k)`, reading their embeddings back from the store.
- `InMemoryImageEmbeddingStore.embeddings_of(paths)` and `vectors_at(ids)` on
  every index read a few gallery vectors back without decoding the whole
  gallery.
- `Candidate.array` reads the matched image as an RGB `uint8` array on first
  access and keeps it, and `Candidate.clear_buffer` drops it again.
- `scripts/benchmark_reranking.py` measures plain retrieval, alpha query
  expansion and k-reciprocal re-ranking on the Oxford Flower dataset. The
  results are in the README.

### Performance
Building a store over all 6149 train images of the Oxford Flower dataset, before
and after the batched gallery build, measured with
[`a benchmark script`](changelog_files/benchmark_store_batching.py) on
the CPU with `PYVISIM_NUM_THREADS=4`. Only the store constructor is timed.

```python
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.neural_networks import ClipEmbedder

train_dataset = OxfordFlowerDataset()
train_image_paths = train_dataset.image_paths

embedder = ClipEmbedder()

image_store = InMemoryImageEmbeddingStore(
    image_paths=train_image_paths,
    embedder=embedder,
    search_index="hnsw",
    index_params={"m": 16, "ef_construction": 200},
)
```

| Before | After |
|---|---|
| 338 s | 249 s |

```python
from pyvisim.classic import FisherVectorEmbedder
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.image_store import InMemoryImageEmbeddingStore

train_dataset = OxfordFlowerDataset()
train_image_paths = train_dataset.image_paths

# Fitted beforehand with learn(images, dim_reduction_factor=2)
embedder = FisherVectorEmbedder(n_components=32)

image_store = InMemoryImageEmbeddingStore(
    image_paths=train_image_paths,
    embedder=embedder,
    search_index="hnsw",
    index_params={"m": 16, "ef_construction": 200},
)
```

| Before | After |
|---|---|
| 2936 s | 2706 s |

### Changed
- ⚠️ `Candidate` is a frozen dataclass instead of a named tuple, so it no longer
  unpacks or indexes: read `candidate.path` and `candidate.score`. It lives in
  `pyvisim.image_store.candidate` and is still exported from `pyvisim.image_store`.
- Added backbones resnet34, resnet50, resnet101, resnet152 under
`pyvisim.neural_networks.backbones`.
- ⚠️ `HnswIndex` takes `graph_degree`, `build_candidates` and
  `search_candidates` instead of `m`, `ef_construction` and `ef_search`, and
  exposes them under those names. The old names are gone, in `index_params`
  too, so a store saved by an earlier release cannot be loaded by this one.
- Every `save_to_disk` rejects a destination whose directory does not exist
  with an `OSError`, and every `load_from_disk` reports a missing file as a
  `FileNotFoundError`. The embedders reported both as a safetensors error.
- ⚠️ `SSIM` and `MSSSIM` score 16 image pairs per batch instead of two, and
  the neural embedders now embed 16 images per forward pass instead of the
  whole input at once. Pass `batch_size=-1` for the previous behaviour.
- ⚠️ `PSNR` takes `batch_size=-1` instead of `batch_size=None` to score the
  whole input as one batch, and defaults to `16` rather than to the whole input.
- ⚠️ `PSNR` raises on a batch holding no image instead of returning an empty
  score matrix, which is what every other metric already did.
- ⚠️ The batch size is a required key of the `.embedder` format, so a file
  written by an earlier release cannot be loaded by this one.

### Removed
- ⚠️ `DeepConvFeature` no longer appends normalized `(x, y)` coordinates to its
  descriptors: the `spatial_embedding` argument is gone and `output_dim` is now
  the channel count of the selected conv layer.
- ⚠️ `pyvisim.serialization.save_embedder_state` and `load_embedder_state`.
  They only called `save_state`/`load_state` with the embedder metadata key,
  which is now exported as `EMBEDDER_METADATA_KEY`.

## [0.9.2] - 2026-08-26

### Added
- `HnswIndex` and `BruteForceIndex` (in `pyvisim.image_store`): an approximate
  HNSW graph and an exhaustive scan, both compiled into the package and built
  in cosine space by default.
- `ExternalSearchIndex` (in `pyvisim.image_store`): searches through an index
  built elsewhere. `ExternalSearchIndex.from_faiss_index(index, vectors=None)`
  adapts any FAISS index without FAISS being a dependency of this library.
- `InMemoryImageEmbeddingStore.retrieve_top_k_similar` ranks the gallery against
  query images and returns `Candidate` matches, both now owned by the store.
- `InMemoryImageEmbeddingStore.save_to_disk` takes the gallery `vectors` to
  write, for an index that hands back an approximation of what it was given.
- `InMemoryImageEmbeddingStore.load_from_disk` forwards keyword arguments:
  `search_index=...` restores a store onto a rebuilt external index, and
  anything else reaches the embedder.

### Changed
- ⚠️ The store's `index_type` parameter is now `search_index`, which takes
  `"hnsw"`, `None` for a brute-force scan, or an `ExternalSearchIndex`. Its
  `quantizer` parameter is now `space`, taking `"cosine"` (the default), `"l2"`
  or `"ip"`.
- ⚠️ The scores of `Candidate` and `search` are distances for the built-in
  indexes, so lower is more similar. An `ExternalSearchIndex` reports whatever
  its own metric produces.
- The index owns the gallery vectors and the store keeps no second copy, so
  `store.embeddings` is read-only. Both built-in indexes decode it out of their
  own storage, which makes every access a fresh copy.
- Store files are written in a new layout; a store saved by an earlier version
  cannot be loaded by this one.

### Removed
- ⚠️ `pyvisim.retrieval` (`ImageRetriever`, `ImageIndex` and the FAISS-backed
  IVF indexes) and `pyvisim.functional`. The store now covers both.
- ⚠️ The `search` extra, along with the `faiss-cpu` dependency behind it.

## [0.9.1] - 2026-08-24

### Added
- Added `Triplet Neural Network` under `pyvisim.neural_networks` with `TripletLoss`.

## [0.9.0] - 2026-08-23

### Fixed
- `PSNR.similarity_score` accepts a channel-less grayscale image again: a 2-D
  array passed with the default `dims="HWC"` raised instead of being read as
  single-channel, unlike `SSIM` and the rest of the library.
- `make test-types` no longer prints a `DeprecationWarning`: the
  `numpy.typing.mypy_plugin` entry is removed from the mypy configuration.
- The development interpreter is pinned to Python 3.10 (`.python-version`), the
  project's minimum supported version and the one every CI job already uses.
- The `ruff check` CI step no longer fails on import sorting (`I001`) in
  `tests/neural_networks/test_oxford_flowers_quick.py` and
  `test_oxford_flowers_slow.py`.

### Added
- `load_from_disk` now forwards keyword arguments to `from_dict`, so an
  embedder can be handed the objects its file cannot hold. The Siamese
  networks use it for their transform:
  `ContrastiveSiameseNetwork.load_from_disk(path, transform=transform)`
  restores the exact embeddings of a network built with a custom one.
- The embedders of `pyvisim.neural_networks` (`ClipEmbedder`,
  `ContrastiveSiameseNetwork`, `BCESiameseNetwork`) are now serialisable to a
  safetensors `.embedder` file via `save_to_disk`/`load_from_disk`, weights
  included; a reloaded embedder produces identical embeddings without downloading
  any pretrained weights.
- `NeuralImageEmbedder` (in `pyvisim.neural_networks`): the shared base for the
  neural embedders, both a `SerializableImageEmbedder` and a `torch.nn.Module`.
  `SiameseNetworkBase` now derives from it, so the Siamese networks and the
  classic embedders expose the same `embed`/`similarity_score` surface.
- Clustering models can now be built from a fitted scikit-learn estimator:
  `KMeans.from_sklearn`, `DiagCovarGaussianMixture.from_sklearn` and
  `PCA.from_sklearn`, plus `load_clustering_model_from_sklearn` on `VLADEncoder`
  and `FisherVectorEncoder` to drop one straight into an encoder. Handy for reusing
  a vocabulary you already trained with scikit-learn.

### Changed
- The Siamese networks now store the `repr` of their transform in the `.embedder`
  file and warn on `load_from_disk` when the rebuilt network's transform differs
  from it, instead of warning on every save of a custom transform.
- `tqdm` and `requests` are no longer runtime dependencies of the core
  package; they moved into the `nn` extra. Only `pyvisim.datasets` uses them,
  and that module already requires `torch` from the same extra.
- CI restores the Oxford Flowers dataset and the pretrained backbone weights from
  the GitHub Actions cache instead of re-downloading them on every run; the new
  `Warm asset cache` workflow keeps that cache populated on `main`.
- The `similarity_func` registry in `pyvisim._utils` now maps the metric names
  straight onto `pyvisim.distance`.
- ℹ️ Dropped scikit-learn as a runtime dependency.
- Added PSNR (under `pyvisim.pixelwise`) and SSIM/MSSSIM (under
`pyvisim.structural`) metrics as well as their benchmark scripts against
existing implementations under `docs/pixelwise/benchmarks` and
`docs/structural/benchmarks`.
- `BCESiameseNetwork` (in `pyvisim.neural_networks`): the pair-classifying
  Siamese variant of Koch, Zemel & Salakhutdinov (2015).
- The Siamese networks are split along a shared abstract base,
  `SiameseNetworkBase`.
- Removed the Siamese Network's train scripts. This is now demonstrated
in a notebook in the "examples" repository.

### Breaking
- ⚠️ `VLADEmbedder`, `FisherVectorEmbedder` and `Pipeline` moved from
  `pyvisim.encoders` to `pyvisim.classic`.
- ⚠️ The bundled pretrained VLAD and Fisher Vector encoders are removed to make the binary smaller, together
  with `from_pretrained`, `PretrainedVLAD`/`PretrainedFisher`, the deprecated
  `weights=` argument and `KMeansWeights`/`GMMWeights`. Train a vocabulary with
  `learn()` and persist it with `save_to_disk`/`load_from_disk` instead.
- ⚠️ "Encoder" is now "embedder" throughout: `ImageEncoderBase` -> `ImageEmbedderBase`,
  `VLADEncoder` -> `VLADEmbedder`, `FisherVectorEncoder` -> `FisherVectorEmbedder`,
  the `Encoder` protocol -> `Embedder`, `encode()` -> `embed()` and `store.encoder`
  -> `store.embedder`.
- ⚠️ Saved models use the `.embedder` suffix and an `embedder_class` state key, so
  existing `.encoder` files no longer load. Re-save them with `save_to_disk`.
- ⚠️ `SiameseNeuralNetwork` is renamed to `ContrastiveSiameseNetwork`
  (`pyvisim.neural_networks.siamese.siamese_neural_network` is gone; the base
  class now lives in `pyvisim.neural_networks.siamese._base_siamese`):

  ```python
  from pyvisim.neural_networks import ContrastiveSiameseNetwork

  model = ContrastiveSiameseNetwork(backbone="resnet18", embedding_dim=128)
  score = model.similarity_score(image1, image2)  # cosine similarity in [-1, 1]
  ```
- ⚠️ The clustering models (`KMeans`, `DiagCovarGaussianMixture`, `PCA`,
  `ClusteringModelBase`) are now internal to the encoders package and moved from
  `pyvisim.clustering` to `pyvisim.classic._clustering`.
- ⚠️ Encoder clustering parameters changed: pass `rng` instead of `random_state`
  inside `kmeans_params` / `gmm_params` / `pca_params` (see
  [vlad.md](docs/classic/vlad.md) and
  [fisher_vector.md](docs/classic/fisher_vector.md) for every accepted key).

## [0.8.2]

### Added
- `SIFT` now exposes the full set of detector parameters (`upsampling`, `n_octaves`,
  `n_scales`, `sigma_min`, `c_dog`, `c_edge`, `n_hist`, `n_ori`, ...) as constructor
  arguments, along with the underlying detector API (`detect`, `extract`,
  `detect_and_extract` and the `keypoints`/`descriptors`/`positions`/... attributes).
  `output_dim` is now `n_hist**2 * n_ori` (still `128` with the defaults).

### Changed
- ℹ️ Removed `OpenCV` and `torchaudio` from the dependency list.
- `SIFT` and `RootSIFT` no longer call OpenCV's `cv2.SIFT`; they now run the pure
  NumPy/Cython SIFT implementation vendored from scikit-image
  (`pyvisim/features/_vendored/sift/`, compiled via `make build-ext`).
  `RootSIFT` subclasses `SIFT` and only adds the Hellinger-kernel normalization.
  With OpenCV gone, `opencv-python-headless` is removed from the dependencies;
  `scipy` returns as a direct dependency (the vendored implementation uses.


### Breaking
-  Some small numerical changes are expected compared to before regarding the `SIFT`
  and `RootSIFT` comoutation are expected due to the migration. For the user, no
  difference in API is observed since only the backend behind these 2 classes change.

## [0.8.1]

### Added
- Structural similarity metrics (in `pyvisim.structural`): `SSIM` (Wang et al., 2004) and the
  multi-scale `MSSSIM` (Wang et al., 2003), computed by a compiled multithreaded Cython
  kernel (thread count via `num_workers` or `PYVISIM_NUM_THREADS`) and matching
  scikit-image / torchmetrics respectively. Both score two image batches into an `(N, M)`
  similarity matrix and take a `batch_size` parameter to bound peak memory (`-1` scores the
  whole input as one batch):

  ```python
  from pyvisim.structural import MSSSIM, SSIM

  scores = SSIM().similarity_score(image1, image2)          # (N, M) matrix in [-1, 1]
  scores = MSSSIM(batch_size=16).similarity_score(gallery, queries)
  ```
- New `pyvisim.distance` module with pyvisim's own pure-NumPy pairwise metrics:
  `cosine_similarity`, `euclidean_distances` and `manhattan_distances`. They keep
  scikit-learn's numerical tricks (float64 upcast and the dot-product expansion for
  Euclidean, zero-safe norms divided out of the result in place for cosine, chunked
  broadcasting with a configurable `working_memory_bytes` budget for Manhattan) and
  are verified against the scikit-learn reference in the test suite, including
  `slow`-marked stress tests on a 100000 x 10000 gallery (size overridable via
  `PYVISIM_TEST_LARGE_ROWS` / `PYVISIM_TEST_LARGE_FEATURES`).

### Changed
- The distance metrics behind `similarity_func` no longer wrap
  `sklearn.metrics.pairwise`; they now resolve to the implementations in
  `pyvisim.distance`. Same names, same results.
- Rolled out lib's own `.mat` loader to replace `scipy.io.loadmat`, so that
the `scipy` dependency could be dropped completely. Added test to verify
that the new loader loads the same data as `scipy.io.loadmat` on the Oxford-102 Flowers dataset.
- `read_image_rgb` in `_utils` now uses `Pillow` to open instead of `cv2.imread` as plan
to be as little dependent on OpenCV as possible.
- CLIP moved from `pyvisim.classic` into `pyvisim.neural_networks` and dropped the
  open_clip dependency entirely. The new `ClipEmbedder` runs pyvisim's own implementation
  of the CLIP image towers (Vision Transformer and modified ResNet) and loads pretrained
  safetensors weights from the Hugging Face Hub — verified numerically equivalent to
  open_clip's image embeddings. Variant names and pretrained tags follow open_clip:
  67 (variant, tag) combinations across 30 variant names are supported (every open_clip
  variant with a standard CLIP image tower and open_clip-format safetensors on the Hub),
  from `RN50` and `ViT-B-32` up to `ViT-g-14`/`ViT-bigG-14`, with weights by OpenAI,
  LAION, DataComp and Meta (MetaCLIP, incl. MetaCLIP-2 worldwide). Enumerate them with
  `pyvisim.neural_networks.clip.available_variants()` / `available_pretrained(variant)`.
  Only the image tower is loaded, always in `float32`; QuickGELU-trained checkpoints
  (like all `"openai"` ones) automatically get the QuickGELU activation. Downloads are
  integrity-checked by huggingface_hub and land in the standard Hugging Face cache
  (`~/.cache/huggingface/hub`), so weights already pulled via open_clip's Hub downloads
  are reused.

  ```python
  from pyvisim.neural_networks import ClipEmbedder

  embedder = ClipEmbedder("ViT-B-32", pretrained="openai")  # "ViT-B/32" works too
  embeddings = embedder.embed(images)  # (num_images, 512); L2-normalized by default
  score = embedder.similarity_score(image1, image2)
  ```

### Breaking
- ⚠️ `CLIPEncoder` is gone. Use `ClipEmbedder` instead: the method is `embed()` (like
  `SiameseNeuralNetwork`), not `encode()`, and it takes open_clip-style `variant` and
  `pretrained` arguments (`ClipEmbedder("ViT-B-32", pretrained="openai")`). CLIP
  `.encoder` files can no longer be loaded; just construct the embedder with the
  variant you want.
- ⚠️ The `nn` extra no longer installs `open_clip_torch`; it now installs
  `huggingface_hub` (for the checkpoint downloads) instead. If your own code imports
  `open_clip`, install it yourself.

## [0.8.0] - 2026-07-04

### Added
- Siamese network for image similarity (in `pyvisim.neural_networks`), replacing the earlier
  sketch. `SiameseNeuralNetwork` wraps a ResNet-18 backbone plus a projection head and hands
  back L2-normalized embeddings, so you can score two images with `similarity_score` or pull the
  raw vectors with `embed`:

  ```python
  from pyvisim.neural_networks import SiameseNeuralNetwork

  model = SiameseNeuralNetwork(backbone="resnet18", embedding_dim=128)
  score = model.similarity_score(image1, image2)  # cosine similarity in [-1, 1]
  ```

  Fine-tune it on labelled pairs with `ContrastiveLoss` (from `pyvisim.neural_networks.losses`),
  or just run the bundled Oxford Flowers training script:
  `python -m pyvisim.neural_networks.scripts.train_siamese_neural_network`. Needs the `nn` extra
  (`pip install "pyvisim[nn]"`).

## [0.7.0] - 2026-07-03

### Added
- `CLIPEncoder` (in `pyvisim.classic`): a pretrained-CLIP image encoder built on
  open_clip. It maps an image straight to a CLIP embedding, so there's no feature
  extractor, clustering model, or `learn` step. Embeddings are L2-normalized by default,
  and it plugs into the usual `similarity_score` / `save_to_disk` / `load_from_disk`
  machinery.

  ```python
  from pyvisim.classic import CLIPEncoder

  clip = CLIPEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
  embeddings = clip.encode(images)
  ```

  Saving stores only the model identifiers (`model_name`, `pretrained` tag, etc.), not
  the weights, so `.encoder` files stay tiny and open_clip re-fetches the weights on load.
- `nn` optional extra (`pip install "pyvisim[nn]"`) now pulls in the whole deep-learning
  stack: `torch`, `torchvision`, `torchaudio` and `open_clip_torch`. It covers
  `DeepConvFeature` (VGG16 deep features), `CLIPEncoder`, and the `datasets` and
  `neural_networks` modules. Everything is imported lazily, so importing `pyvisim` never
  requires it; you only hit the error (with an install hint) the first time you actually
  build one of these without it installed.
- `search` optional extra (`pip install "pyvisim[search]"`) that pulls in `faiss-cpu` for
  the retrieval / image-store stack: `InMemoryImageEmbeddingStore`, `ImageRetriever` and
  the `ImageIndex*` classes. faiss is imported lazily too, so you only need it when you
  build a store or an index.

### Breaking
- ⚠️ `pip install pyvisim` no longer installs torch or faiss. The base install now covers
  the SIFT/RootSIFT encoders only. Install `[nn]` for deep features and CLIP, `[search]`
  for the image store and retrieval, or `pip install "pyvisim[nn,search]"` for everything.
  Heads up: the VGG16 pretrained encoders (`OXFORD102_K256_VGG16*`) build a
  `DeepConvFeature`, so they now need the `nn` extra.

## [0.6.0] - 2026-06-20

### Added
- `InMemoryImageEmbeddingStore` (in `pyvisim.image_store`): the new gallery object.
  Give it image paths, an encoder, and an index type, and it encodes everything,
  builds a FAISS index, and searches itself:

  ```python
  from pyvisim.image_store import InMemoryImageEmbeddingStore

  store = InMemoryImageEmbeddingStore(
      gallery_paths, encoder, "ivf-flat",
      quantizer="inner_product", index_params={"nlist": 100, "nprobe": 8},
  )
  results = store.retrieve_top_k_similar(query_images, k=5)
  ```

  It saves to a single `.safetensors` file (embeddings, paths, index config and the
  fully serialised encoder) and `load_from_disk` rebuilds it without re-encoding.
- `index_type` strings select the index structure: `"ivf-flat"` and `"ivf-pq"` work
  today; `"hnsw"` and `"int8"` are sketched for a future release and raise
  `NotImplementedError` for now.
- Encoders and `Pipeline` gained `to_dict`/`from_dict`, and there's a new
  `EmbeddingStore` protocol in `pyvisim.typing`.

### Changed
- `retrieve_top_k_similar(query_images, store, k=5)` now takes a store and searches
  through its index. `top_k_map` and `top_k_accuracy` take a store too, instead of a
  separate `(encoding_map, encoder)` pair.
- `ImageRetriever` now wraps a store: `ImageRetriever(store)`.
- The image indexes take the gallery as `(paths, vectors)` rather than a mapping, and
  the trained FAISS index is now the single owner of the vectors. Read them back with
  `index.reconstruct()` (or `store.embeddings`) instead of keeping a second copy.

### Breaking
- ⚠️ `ImageEncodingMap` is gone. Build an `InMemoryImageEmbeddingStore` from your image
  paths instead of a `{path: vector}` mapping.
- ⚠️ `Encoder.generate_encoding_map(...)` and `Pipeline.generate_encoding_map(...)` are
  removed. Pass the paths straight to `InMemoryImageEmbeddingStore`.
- ⚠️ `retrieve_top_k_similar` dropped its `dataset`/`encoder`/`index` arguments (and the
  brute-force path); pass a store. The same applies to `top_k_map`/`top_k_accuracy`.

## [v0.5.1] - 2026-06-19

### Fixed
- The method `_from_config` of `DeepConvFeature` was using the deprecated
  `model` argument instead of `backbone`. This version only fixed that.

## [v0.5.0] - 2026-06-19

### Added
- New `pyvisim.retrieval` package for fast similarity search. Wrap an
  `ImageEncodingMap` in an index (`ImageIndexIVFFlat` or `ImageIndexIVFPQ`,
  both `l2` or `inner_product`), then hand it to an `ImageRetriever`:

  ```python
  from pyvisim.retrieval import ImageIndexIVFFlat, ImageRetriever

  index = ImageIndexIVFFlat(encoding_map, quantizer="inner_product", nlist=100)
  retriever = ImageRetriever(index)
  results = retriever.retrieve_top_k_similar(query_images, k=5)
  ```
- New `pyvisim.functional` module holding `retrieve_top_k_similar` and the
  `Candidate(path, score)` result type.

### Changed
- `retrieve_top_k_similar` now ranks a whole batch of query images in one shot
  and returns one ranked `list[Candidate]` per query (in input order), so a
  single call can search many images at once. Pass an `index=` to run the search
  through FAISS instead of brute-force cosine.

### Breaking
- ⚠️ `retrieve_top_k_similar` moved out of `pyvisim.eval` into
  `pyvisim.functional`. Update your imports:
  `from pyvisim.functional import retrieve_top_k_similar`.
- ⚠️ Its return type changed from `list[tuple[str, float]]` to
  `list[list[Candidate]]` (one list per query image). Read `candidate.path` and
  `candidate.score` instead of unpacking a tuple.
- ⚠️ The getters for the `pca` and `clustering_model` attributes of all encoders
   are removed (the attributes are now read-only). This is in order to discourage
   users from mutating the clustering internals, which could break the
   algorithm completely. Also, once trained, there's not really a reason to
   have to mutate those models at all because any different model would be
   basically wrong for the trained encoder.
- ⚠️ The `ImageEncodingMap` does not take the `Encoder` as an argument anymore.

## [v0.4.1] - 2026-06-18

### Added
- `DeepConvFeature` now takes a `backbone` argument. Pass `"vgg16"` to grab a
  torchvision VGG16 with ImageNet weights, or hand it your own `torch.nn.Module`.
  Leave it out and you still get the default VGG16.

### Deprecated
- The `model` argument of `DeepConvFeature` is deprecated; use `backbone`
  instead. If you still pass `model`, it's used as the backbone and you'll get a
  `DeprecationWarning`. It'll be removed in a future release.

## [v0.4.0] - 2026-06-18

### Added
- `from_pretrained()` on `VLADEncoder` and `FisherVectorEncoder`, plus the
  `PretrainedVLAD` and `PretrainedFisher` enums. Pick a bundled encoder and
  you're ready to go: `VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)`.

### Changed
- Encoders now serialize to a single safetensors `.encoder` file that captures
  everything: the clustering model, PCA, normalization settings, the feature
  extractor and the similarity metric. `load_from_disk()` takes just the path
  and rebuilds the whole encoder, so there's nothing else to pass back in.
- For a `DeepConvFeature` extractor, the default torchvision model is rebuilt on
  load (only a flag is stored), while a model you supply yourself has its full
  `state_dict` embedded so your trained weights come back exactly.
- `similarity_func` is now chosen by name: `"cosine"` (default), `"euclidean"`,
  `"l1"` or `"manhattan"`.
- The pretrained Oxford-102 weights ship as `.encoder` files instead of `.pkl`,
  shrinking them from ~144 MB to ~12 MB (the K-Means training `labels_` array is
  no longer stored).

### Removed
- Dropped `joblib` entirely in favor of safetensors.
- ⚠️ You can no longer pass your own similarity function; use one of the four
  built-in metric names above.

### Deprecated
- Loading pretrained weights via `KMeansWeights`/`GMMWeights` (the `weights=`
  argument) is deprecated and will be removed in 1.0.0. Use `from_pretrained()`
  or `load_from_disk()` with `.encoder` files instead.

## [v0.3.1] - 2026-06-18

### Changed
- `ImageEncodingMap` now encodes every image up front instead of lazily on first access, which drops the in-memory buffer machinery and simplifies the class.
- `ImageEncodingMap.save_to_disk()` / `load_from_disk()` now use the safetensors format instead of HDF5. Files default to the `.safetensors` extension.
- `skip_errors` moved from `save_to_disk()` to the `ImageEncodingMap` constructor, since encoding now happens at construction time.

### Removed
- Dropped `h5py` as a dependency; added `safetensors`.
- Removed `ImageEncodingMap.clear_buffer()` (there's no buffer to clear anymore).

### Breaking
- ⚠️ Unreadable or missing images now raise (`FileNotFoundError` / `ValueError`) when the map is built, not on first access. Use `skip_errors=True` to drop them with a warning instead.
- ⚠️ Encoding maps saved with `0.3.0` (HDF5) can't be loaded by `0.3.1`; re-save them as safetensors.

## [v0.3.0] - 2026-06-17

### Added
- New encoding map feature for the encoders (#36).
- PyPI publishing step in the CI workflow, so releases ship automatically (#37).

### Changed
- Batches now fetch dynamically from PyPI instead of being hardcoded (#38).
- Moved the notebooks into a tidier layout (#35).

### Fixed
- Dropped the deprecated `project.license` TOML table in `pyproject.toml` (#39).

## [v0.2.0] - 2026-06-16

### Added
- Clustering models with a fresh public API (#19), plus docs to match (#21).
- Public types `ImageInput` and `MatLike`, and you can now pass torch images straight in (#23).
- Unit tests across the board (#24), including behavioral tests that check VLAD and Fisher Vector encoders return the same vector before and after serialization (#32).
- Early sketch of a Siamese neural net (#26).

### Changed
- Migrated tooling to `uv` (#5).
- Added ruff, pre-commit hooks, and a CI pipeline that runs on every PR (#9).
- Integrated mypy and cleaned up the type errors (#8, #13).
- Now compatible with Python 3.10 through 3.12 (#14, #15).
- Refreshed the outdated getting-started notebook (#30).

### Fixed
- `VLADEncoder` now raises if no descriptor is extracted, instead of failing silently (#11).
- Use `flatten()` instead of `squeeze()` for setid arrays, so single-element arrays behave (#29).

## [v0.1.3-alpha] - 2025-01-24

- Initial alpha release.
