# pyvisim.typing

File: [`pyvisim/typing/__init__.py`](../pyvisim/typing/__init__.py)
(Implementation: numeric types and image normalization live in
[`pyvisim/typing/numeric.py`](../pyvisim/typing/numeric.py); the embedder protocol in
[`pyvisim/typing/embedders.py`](../pyvisim/typing/embedders.py))

`pyvisim.typing` is the public home for the input types and normalization helpers
used across the library. Everything that accepts image data uses the types defined
here so you only need to learn them once.

## Types

### `MatLike`

```python
MatLike = np.ndarray | torch.Tensor | npt.ArrayLike
```

Anything that can be treated as a numerical image array:

- a NumPy `ndarray` — the library's internal representation
- a PyTorch `Tensor` — converted to NumPy automatically before the feature extractor sees it
- anything `numpy.asarray` can turn into a numeric array: nested lists of numbers, objects
  with `__array__`, etc.

### `ImageInput`

```python
ImageInput = MatLike | Iterable[MatLike]
```

The widest input type accepted by `embed`, `learn`, and `similarity_score`. You can pass:

- a single image (NumPy array, tensor, ...)
- a single *batched* array where one axis is a batch dimension (use `dims` to say which)
- an iterable of individual images — e.g. a generator over large datasets

### `Embedder`

```python
class Embedder(Protocol):
    def embed(self, images: ImageInput, *, dims=..., value_range=...) -> FloatNumpyArray: ...
```

A structural type (a `typing.Protocol`) for anything that turns images into vectors: all
it needs is an `embed` method. `VLADEmbedder`, `FisherVectorEmbedder`, and `Pipeline` all
satisfy it without inheriting from it. That's what lets
[`InMemoryImageEmbeddingStore`](image_similarity_retrieval/image_store.md) accept any of them without importing the
concrete embedder classes.

There's a matching `EmbeddingStore` protocol too (the gallery surface that retrieval and
evaluation rely on: `paths`, `embeddings`, `embedder`, and `search`).
`InMemoryImageEmbeddingStore` satisfies it structurally, so `top_k_map` and
`top_k_accuracy` stay decoupled from the concrete store. `SearchIndex` describes the
other half, what a store searches through: the `vectors` an index owns, its `dim`,
`vectors_at` to read a few of those vectors back by row number, and its own `search`.

## The `dims` string

Every method that accepts image data also accepts a `dims` keyword argument. It's a short
string that tells the library how to read your array's axes, one character per dimension
in the exact order the axes appear:

| Character | Axis |
|-----------|------|
| `"H"` | height (rows) |
| `"W"` | width (columns) |
| `"C"` | channels (e.g. RGB) |
| `"B"` | batch size |

`"H"` and `"W"` are mandatory; `"C"` and `"B"` are optional. The default is `"HWC"` —
which is the standard NumPy/OpenCV single-image layout. Common layouts:

| `dims` | Shape meaning | Typical source |
|--------|---------------|----------------|
| `"HWC"` | height x width x channels | NumPy / OpenCV (**default**) |
| `"CHW"` | channels x height x width | PyTorch single image (`torch.Tensor`) |
| `"BHWC"` | batch x height x width x channels | NumPy batch |
| `"BCHW"` | batch x channels x height x width | PyTorch batched `Tensor` |
| `"HWCB"` | height x width x channels x batch | some data loaders |
| `"HW"` | height x width only (grayscale) | grayscale images |

When `"B"` is present the batch is automatically split so every image is processed
individually. You don't need to loop yourself.

`dims` is **case-insensitive**: `"hwc"`, `"HWC"`, `"Hwc"` all work the same way.

## The `value_range` tuple

```python
value_range: tuple[float, float] = (0.0, 255.0)  # default
```

Tells the library what numerical range your input values live in. Pixels are rescaled
into `[0, 255]` before feature extraction. Common cases:

| Your image | `value_range` to pass |
|------------|-----------------------|
| `uint8` NumPy array, values 0–255 | nothing — this is the default |
| float tensor from `torchvision.transforms.ToTensor()`, values 0–1 | `(0.0, 1.0)` |
| float image, values −1 to 1 (e.g. some augmentation pipelines) | `(-1.0, 1.0)` |

If your image is already `uint8` in `(0, 255)` the rescaling step is a no-op.

## Passing torch images to an embedder

You don't need to convert manually. Just tell the embedder what layout your tensor is in:

```python
import torch
from pyvisim.classic import VLADEmbedder

embedder = VLADEmbedder(n_clusters=64)
embedder.learn(train_images)  # train_images: list of uint8 HWC NumPy arrays

# PyTorch DataLoader typically yields BCHW float tensors in [0, 1]
batch: torch.Tensor  # shape (8, 3, 224, 224)
embeddings = embedder.embed(batch, dims="BCHW", value_range=(0.0, 1.0))
# embeddings.shape == (8, 64 * feature_dim)
```

The conversion happens once per call, before the feature extractor runs, so there is no
overhead from calling `to_single_image` yourself first.
