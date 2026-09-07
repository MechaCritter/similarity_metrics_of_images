"""Tests for :class:`pyvisim.image_store.InMemoryImageEmbeddingStore`."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PIL import Image

from pyvisim.classic import Pipeline, VLADEmbedder
from pyvisim.image_store import (
    BruteForceIndex,
    Candidate,
    ExternalSearchIndex,
    HnswIndex,
    InMemoryImageEmbeddingStore,
)

# The reading stage is internal to the store, which exposes no way to run it
# without embedding. Driving it directly is what keeps the embedder out of the
# measurement.
from pyvisim.image_store.image_store import _decoded_images
from pyvisim.neural_networks import ContrastiveSiameseNetwork

#: Images the reading measurement decodes.
_TIMED_GALLERY_SIZE = 24
#: Side length of those images, sized so that decoding one costs far more than
#: the thread hand-off around it.
_TIMED_IMAGE_SIDE = 1024
#: Times each worker count is measured.
_TIMING_ROUNDS = 3
#: Speed-up two reading threads must reach over a single one.
_MIN_SPEED_UP = 1.5


@pytest.fixture(scope="module")
def gallery_paths(
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> list[str]:
    """Write the training images to disk and return their paths.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :param category_train_images_flat: corner-rich training images.
    :returns: one ``.png`` path per training image.
    """
    directory = tmp_path_factory.mktemp("store_gallery")
    paths: list[str] = []
    for index, image in enumerate(category_train_images_flat):
        rgb = np.stack([image, image, image], axis=-1)
        path = directory / f"img_{index}.png"
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def store(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
) -> InMemoryImageEmbeddingStore:
    """An :class:`InMemoryImageEmbeddingStore` over the on-disk gallery.

    :param gallery_paths: the gallery image paths.
    :param learned_vlad_embedder: a fitted VLAD embedder (PCA and non-PCA variants).
    :returns: a store backed by an exact brute-force index.
    """
    return InMemoryImageEmbeddingStore(gallery_paths, learned_vlad_embedder)


@pytest.fixture(scope="module")
def hnsw_store(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
) -> InMemoryImageEmbeddingStore:
    """An :class:`InMemoryImageEmbeddingStore` on an HNSW graph.

    :param gallery_paths: the gallery image paths.
    :param learned_vlad_embedder: a fitted VLAD embedder.
    :returns: a store backed by an approximate HNSW index.
    """
    return InMemoryImageEmbeddingStore(
        gallery_paths, learned_vlad_embedder, "hnsw", index_params={"graph_degree": 8}
    )


@pytest.fixture(scope="module")
def timed_gallery_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Write a gallery whose images are slow enough to decode to be timed.

    The images of the other fixtures decode in microseconds, which is drowned
    out by the cost of handing them between threads. These are large and noisy,
    so the decoding is what a measurement over them observes.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :returns: one ``.jpg`` path per image.
    """
    directory = tmp_path_factory.mktemp("store_timing_gallery")
    rng = np.random.default_rng(0)
    paths: list[str] = []
    for index in range(_TIMED_GALLERY_SIZE):
        noise = rng.integers(
            0, 256, size=(_TIMED_IMAGE_SIDE, _TIMED_IMAGE_SIDE, 3), dtype=np.uint8
        )
        path = directory / f"img_{index}.jpg"
        Image.fromarray(noise).save(path, quality=95)
        paths.append(str(path))
    return paths


def _warm_page_cache(paths: list[str]) -> None:
    """Read every file once, so that no measurement pays for the first disk read."""
    for path in paths:
        with open(path, "rb") as handle:
            handle.read()


def _read_seconds(paths: list[str], num_workers: int) -> float:
    """Read and decode the whole gallery once, and return how long it took.

    :param paths: gallery image paths to read.
    :param num_workers: threads decoding the image files.
    :returns: the wall-clock duration of the read, in seconds.
    """
    failures: list[str] = []
    start = time.perf_counter()
    for _ in _decoded_images(paths, num_workers, len(paths), False, failures):
        pass
    return time.perf_counter() - start


# Construction and exposed state


def test_reports_paths_dim_and_len(
    store: InMemoryImageEmbeddingStore, gallery_paths: list[str]
) -> None:
    """The store exposes the gallery paths, dimensionality and size."""
    assert store.paths == gallery_paths
    assert len(store) == len(gallery_paths)
    assert store.dim == store.embeddings.shape[1]
    assert store.index_name == "brute-force"
    assert store.space == "cosine"


def test_contains_and_repr(store: InMemoryImageEmbeddingStore) -> None:
    """``in`` checks gallery membership and ``repr`` names the store."""
    assert store.paths[0] in store
    assert "absent.png" not in store
    assert "InMemoryImageEmbeddingStore(" in repr(store)


def test_default_store_builds_a_brute_force_index(
    store: InMemoryImageEmbeddingStore,
) -> None:
    """Leaving ``search_index`` out scans the gallery exhaustively."""
    assert isinstance(store.index, BruteForceIndex)


def test_hnsw_store_builds_a_graph_index(
    hnsw_store: InMemoryImageEmbeddingStore,
) -> None:
    """``"hnsw"`` builds a graph index with the parameters it was given."""
    assert isinstance(hnsw_store.index, HnswIndex)
    assert hnsw_store.index.graph_degree == 8
    assert hnsw_store.index_name == "hnsw"
    assert hnsw_store.index_params == {"graph_degree": 8}


def test_embeddings_come_from_the_index(store: InMemoryImageEmbeddingStore) -> None:
    """The store holds no embeddings of its own, only the index does."""
    assert np.array_equal(store.embeddings, store.index.vectors)


def test_embeddings_are_read_only(store: InMemoryImageEmbeddingStore) -> None:
    """The gallery the store hands out cannot be written to."""
    with pytest.raises(ValueError, match="read-only"):
        store.embeddings[0, 0] = 1.0


def test_hnsw_store_copies_its_embeddings_per_access(
    hnsw_store: InMemoryImageEmbeddingStore,
) -> None:
    """A graph-backed store decodes a fresh matrix on every access."""
    assert hnsw_store.embeddings is not hnsw_store.embeddings
    assert np.array_equal(hnsw_store.embeddings, hnsw_store.embeddings)


def test_cosine_store_normalizes_its_embeddings(
    store: InMemoryImageEmbeddingStore,
) -> None:
    """A cosine store exposes L2-normalised embeddings."""
    norms = np.linalg.norm(store.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_store_keeps_the_raw_embeddings(
    gallery_paths: list[str], learned_vlad_embedder: VLADEmbedder
) -> None:
    """An L2 store indexes the vectors as the embedder produced them."""
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:6], learned_vlad_embedder, space="l2"
    )
    norms = np.linalg.norm(store.embeddings, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_search_returns_expected_shapes(store: InMemoryImageEmbeddingStore) -> None:
    """``search`` returns ``(M, k)`` score and id arrays."""
    scores, ids = store.search(store.embeddings[:3], k=4)
    assert scores.shape == (3, 4)
    assert ids.shape == (3, 4)


def test_retrieve_recovers_self(
    store: InMemoryImageEmbeddingStore,
    gallery_paths: list[str],
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Retrieving with a gallery image returns that image as the top match."""
    gray = category_train_images_flat[2]
    probe = np.stack([gray, gray, gray], axis=-1)
    results = store.retrieve_top_k_similar(probe, k=3)
    assert isinstance(results[0][0], Candidate)
    assert results[0][0].path == gallery_paths[2]


def test_retrieve_ranks_every_query(
    store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A batch of queries yields one ranked list each, in input order."""
    probes = [
        np.stack([gray, gray, gray], axis=-1) for gray in category_train_images_flat[:3]
    ]
    results = store.retrieve_top_k_similar(probes, k=4)
    assert len(results) == 3
    assert all(len(ranked) == 4 for ranked in results)
    assert all(candidate.path in store for ranked in results for candidate in ranked)


def test_retrieve_drops_the_missing_neighbours(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Asking for more matches than the gallery holds returns what there is."""
    store = InMemoryImageEmbeddingStore(gallery_paths[:3], learned_vlad_embedder)
    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    assert len(store.retrieve_top_k_similar(probe, k=10)[0]) == 3


def test_hnsw_store_matches_the_exact_store(
    store: InMemoryImageEmbeddingStore,
    hnsw_store: InMemoryImageEmbeddingStore,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """On this gallery the graph reproduces the exhaustive ranking."""
    gray = category_train_images_flat[1]
    probe = np.stack([gray, gray, gray], axis=-1)
    exact = store.retrieve_top_k_similar(probe, k=5)[0]
    approximate = hnsw_store.retrieve_top_k_similar(probe, k=5)[0]
    assert [c.path for c in approximate] == [c.path for c in exact]


def test_unknown_search_index_raises(
    gallery_paths: list[str], learned_vlad_embedder: VLADEmbedder
) -> None:
    """An unknown search_index is rejected before any image is embedded."""
    with pytest.raises(ValueError, match="Unknown search_index"):
        InMemoryImageEmbeddingStore(gallery_paths, learned_vlad_embedder, "bogus")


@pytest.mark.parametrize("name", ["num_workers", "num_prefetch_batches"])
@pytest.mark.parametrize("value", [0, -1, 2.5])
def test_non_positive_thread_settings_raise(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    name: str,
    value: object,
) -> None:
    """The reading knobs are rejected before any image is embedded."""
    with pytest.raises(ValueError, match=f"'{name}' must be a positive integer"):
        InMemoryImageEmbeddingStore(
            gallery_paths,
            learned_vlad_embedder,
            **{name: value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("num_prefetch_batches", [1, 2, 4])
def test_prefetch_batches_do_not_change_the_gallery(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    store: InMemoryImageEmbeddingStore,
    num_prefetch_batches: int,
) -> None:
    """Reading further ahead is a performance knob, not a result change."""
    prefetching = InMemoryImageEmbeddingStore(
        gallery_paths, learned_vlad_embedder, num_prefetch_batches=num_prefetch_batches
    )
    assert prefetching.paths == store.paths
    np.testing.assert_allclose(prefetching.embeddings, store.embeddings)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_num_workers_no_not_change_gallery(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    store: InMemoryImageEmbeddingStore,
    num_workers: int,
) -> None:
    """Reading on more threads is a performance knob, not a result change."""
    threaded = InMemoryImageEmbeddingStore(
        gallery_paths, learned_vlad_embedder, num_workers=num_workers
    )
    assert threaded.paths == store.paths
    np.testing.assert_allclose(threaded.embeddings, store.embeddings)


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2,
    reason="Decoding on two threads cannot outrun one on a single core.",
)
def test_two_workers_load_faster_than_one(timed_gallery_paths: list[str]) -> None:
    """Two reading threads decode the gallery at least 50% faster than one.

    Only the reading stage is timed: no embedder runs, so the measurement
    covers the stage ``num_workers`` actually controls. The files are read once
    up front, which leaves them in the page cache of the operating system and
    compares how well the decoding parallelises rather than how fast the disk
    is.
    """
    _warm_page_cache(timed_gallery_paths)
    timings: dict[int, list[float]] = {1: [], 2: []}
    for _ in range(_TIMING_ROUNDS):
        for num_workers in timings:
            timings[num_workers].append(_read_seconds(timed_gallery_paths, num_workers))

    single, paired = min(timings[1]), min(timings[2])
    speed_up = single / paired
    assert speed_up >= _MIN_SPEED_UP, (
        f"Reading {len(timed_gallery_paths)} images took {single:.3f}s on one "
        f"thread and {paired:.3f}s on two, a speed-up of only {speed_up:.2f}x."
    )


def test_non_string_path_raises(learned_vlad_embedder: VLADEmbedder) -> None:
    """A non-string path is rejected before any embedding happens."""
    with pytest.raises(TypeError, match="Image paths must be strings"):
        InMemoryImageEmbeddingStore([123], learned_vlad_embedder)  # type: ignore[list-item]


def test_missing_file_raises(
    learned_vlad_embedder: VLADEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A missing image file aborts construction by default."""
    missing = str(tmp_path_factory.mktemp("missing") / "gone.png")
    with pytest.raises(FileNotFoundError):
        InMemoryImageEmbeddingStore([missing], learned_vlad_embedder)


def test_skip_errors_warns_and_keeps_good_images(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``skip_errors`` warns about and omits images that fail to embed."""
    missing = str(tmp_path_factory.mktemp("partial") / "gone.png")
    with pytest.warns(FutureWarning, match="Skipped 1 image"):
        store = InMemoryImageEmbeddingStore(
            [*gallery_paths[:5], missing],
            learned_vlad_embedder,
            skip_errors=True,
        )
    assert store.paths == gallery_paths[:5]


def test_all_images_unreadable_raises(
    learned_vlad_embedder: VLADEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A store cannot be built when every image fails to embed."""
    missing = str(tmp_path_factory.mktemp("empty") / "gone.png")
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="No images could be embedded"):
            InMemoryImageEmbeddingStore(
                [missing], learned_vlad_embedder, skip_errors=True
            )


def test_duplicate_paths_are_dropped(
    gallery_paths: list[str], learned_vlad_embedder: VLADEmbedder
) -> None:
    """A path given twice is embedded and indexed once."""
    store = InMemoryImageEmbeddingStore(
        [gallery_paths[0], gallery_paths[0], gallery_paths[1]], learned_vlad_embedder
    )
    assert store.paths == gallery_paths[:2]


# Saving and loading


def test_save_appends_safetensors_suffix(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The save path gains a ``.safetensors`` suffix when missing."""
    target = tmp_path_factory.mktemp("save_suffix") / "mystore"
    written = store.save_to_disk(target)
    assert written.suffix == ".safetensors"
    assert written.exists()


def test_save_load_preserves_paths(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store keeps the gallery paths in order."""
    target = tmp_path_factory.mktemp("rt_paths") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))
    assert loaded.paths == store.paths


def test_save_load_preserves_index_config(
    hnsw_store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store keeps the index name, space and params."""
    target = tmp_path_factory.mktemp("rt_cfg") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(hnsw_store.save_to_disk(target))
    assert loaded.index_name == hnsw_store.index_name
    assert loaded.space == hnsw_store.space
    assert loaded.index_params == hnsw_store.index_params
    assert isinstance(loaded.index, HnswIndex)


def test_save_load_preserves_embeddings(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store's embeddings match the original exactly."""
    target = tmp_path_factory.mktemp("rt_emb") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))
    assert loaded.embeddings.shape == store.embeddings.shape
    assert np.allclose(loaded.embeddings, store.embeddings, atol=1e-6)


def test_save_accepts_replacement_vectors(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """``vectors`` writes the given embeddings instead of the index's own."""
    target = tmp_path_factory.mktemp("rt_lossy") / "store.safetensors"
    originals = store.embeddings * 3.0
    loaded = InMemoryImageEmbeddingStore.load_from_disk(
        store.save_to_disk(target, vectors=originals), search_index=None
    )
    # The reloaded cosine store re-normalises, so the direction is what survives.
    assert np.allclose(loaded.embeddings, store.embeddings, atol=1e-5)


def test_save_rejects_mismatched_vectors(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """``vectors`` must hold one row per gallery path."""
    target = tmp_path_factory.mktemp("rt_bad_vec") / "store.safetensors"
    with pytest.raises(ValueError, match="one row per"):
        store.save_to_disk(target, vectors=store.embeddings[:2])


def test_save_does_not_mutate_store(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Serialising the store leaves its own state untouched."""
    before = store.embeddings.copy()
    target = tmp_path_factory.mktemp("no_mutate") / "store.safetensors"
    store.save_to_disk(target)
    assert np.array_equal(store.embeddings, before)
    assert store.paths == store.paths


def test_store_with_pipeline_embedder_round_trips(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A store built on a Pipeline serialises and reconstructs the Pipeline."""
    pipeline = Pipeline([learned_vlad_embedder])
    store = InMemoryImageEmbeddingStore(gallery_paths[:8], pipeline)
    target = tmp_path_factory.mktemp("pipeline_store") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))

    assert isinstance(loaded.embedder, Pipeline)
    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    assert np.allclose(
        loaded.embedder.embed(probe), store.embedder.embed(probe), atol=1e-5
    )


def test_save_load_preserves_embedder(
    store: InMemoryImageEmbeddingStore,
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """The reloaded store carries an equivalent embedder.

    The reconstructed embedder is compared against the store's own embedder
    behaviourally (same image embeds to the same vector) and against the same
    embedder serialised on its own with ``save_to_disk``.
    """
    target = tmp_path_factory.mktemp("rt_embedder")
    loaded = InMemoryImageEmbeddingStore.load_from_disk(
        store.save_to_disk(target / "store.safetensors")
    )
    assert isinstance(loaded.embedder, VLADEmbedder)

    # The store's embedder, serialised on its own, reloaded from disk.
    embedder_path = store.embedder.save_to_disk(target / "embedder")
    directly_loaded = VLADEmbedder.load_from_disk(embedder_path)

    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    from_store = store.embedder.embed(probe)
    from_loaded_store = loaded.embedder.embed(probe)
    from_direct = directly_loaded.embed(probe)

    assert np.allclose(from_loaded_store, from_store, atol=1e-5)
    assert np.allclose(from_loaded_store, from_direct, atol=1e-5)


def test_save_load_preserves_a_neural_embedder(
    gallery_paths: list[str],
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A store built on a neural embedder round-trips through disk.

    The neural embedder travels inside the store file with its weights, so
    the reloaded store embeds a probe image exactly like the original one.
    """
    embedder = ContrastiveSiameseNetwork(
        embedding_dim=8, pretrained_backbone=False, similarity_func="cosine"
    )
    store = InMemoryImageEmbeddingStore(gallery_paths[:4], embedder)
    target = tmp_path_factory.mktemp("rt_neural_embedder")
    loaded = InMemoryImageEmbeddingStore.load_from_disk(
        store.save_to_disk(target / "store.safetensors")
    )
    assert isinstance(loaded.embedder, ContrastiveSiameseNetwork)

    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    assert np.array_equal(loaded.embedder.embed(probe), embedder.embed(probe))


def test_load_forwards_kwargs_to_the_embedder(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A keyword argument the embedder does not take is reported."""
    target = tmp_path_factory.mktemp("rt_kwargs") / "store.safetensors"
    written = store.save_to_disk(target)
    with pytest.raises(TypeError):
        InMemoryImageEmbeddingStore.load_from_disk(written, transform=object())


def test_load_missing_file_raises(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Loading a non-existent store file raises ``FileNotFoundError``."""
    missing = tmp_path_factory.mktemp("missing") / "absent.safetensors"
    with pytest.raises(FileNotFoundError):
        InMemoryImageEmbeddingStore.load_from_disk(missing)


def test_load_rejects_a_foreign_file(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A file written by something else is rejected."""
    target = tmp_path_factory.mktemp("foreign")
    embedder_file = store.embedder.save_to_disk(target / "embedder")
    with pytest.raises(ValueError, match="pyvisim_store"):
        InMemoryImageEmbeddingStore.load_from_disk(embedder_file)


def test_load_rejects_a_non_index_search_index(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """``search_index`` must be an :class:`ExternalSearchIndex`."""
    target = tmp_path_factory.mktemp("rt_bad_index") / "store.safetensors"
    written = store.save_to_disk(target)
    with pytest.raises(TypeError, match="must be an ExternalSearchIndex"):
        InMemoryImageEmbeddingStore.load_from_disk(written, search_index="hnsw")


# Stores on an index built elsewhere


def test_external_store_adopts_the_index(
    gallery_paths: list[str], learned_vlad_embedder: VLADEmbedder
) -> None:
    """An external index is searched as-is, and nothing is embedded."""
    faiss = pytest.importorskip("faiss")
    source = InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_embedder)
    vectors = np.ascontiguousarray(source.embeddings)

    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:6],
        learned_vlad_embedder,
        ExternalSearchIndex.from_faiss_index(flat, name="flat-ip"),
    )

    assert store.index_name == "flat-ip"
    assert isinstance(store.index, ExternalSearchIndex)
    assert np.allclose(store.embeddings, vectors, atol=1e-6)
    # An inner-product index ranks the query itself highest, not lowest.
    scores, ids = store.search(vectors[:1], k=3)
    assert ids[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_external_store_rejects_a_path_count_mismatch(
    gallery_paths: list[str], learned_vlad_embedder: VLADEmbedder
) -> None:
    """The index must hold exactly one vector per gallery path."""
    faiss = pytest.importorskip("faiss")
    source = InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_embedder)
    flat = faiss.IndexFlatIP(source.dim)
    flat.add(np.ascontiguousarray(source.embeddings))

    with pytest.raises(ValueError, match="image paths were given"):
        InMemoryImageEmbeddingStore(
            gallery_paths[:4],
            learned_vlad_embedder,
            ExternalSearchIndex.from_faiss_index(flat),
        )


def test_external_store_round_trips_on_a_rebuilt_index(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A store on an external index reloads onto a rebuilt one."""
    faiss = pytest.importorskip("faiss")
    source = InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_embedder)
    vectors = np.ascontiguousarray(source.embeddings)
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:6],
        learned_vlad_embedder,
        ExternalSearchIndex.from_faiss_index(flat, name="flat-ip"),
    )

    target = tmp_path_factory.mktemp("rt_external") / "store.safetensors"
    rebuilt = faiss.IndexFlatIP(vectors.shape[1])
    rebuilt.add(vectors)
    loaded = InMemoryImageEmbeddingStore.load_from_disk(
        store.save_to_disk(target),
        search_index=ExternalSearchIndex.from_faiss_index(rebuilt, name="flat-ip"),
    )

    assert loaded.index_name == "flat-ip"
    assert np.allclose(loaded.embeddings, store.embeddings, atol=1e-6)


def test_load_warns_when_the_index_name_differs(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Reloading onto a differently named index is reported."""
    faiss = pytest.importorskip("faiss")
    source = InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_embedder)
    vectors = np.ascontiguousarray(source.embeddings)
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:6],
        learned_vlad_embedder,
        ExternalSearchIndex.from_faiss_index(flat, name="flat-ip"),
    )
    target = tmp_path_factory.mktemp("rt_renamed") / "store.safetensors"
    written = store.save_to_disk(target)

    with pytest.warns(FutureWarning, match="named 'other'"):
        loaded = InMemoryImageEmbeddingStore.load_from_disk(
            written,
            search_index=ExternalSearchIndex.from_faiss_index(flat, name="other"),
        )
    assert loaded.index_name == "other"


def test_load_without_an_index_falls_back_to_brute_force(
    gallery_paths: list[str],
    learned_vlad_embedder: VLADEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A store saved on an external index reloads onto an exact scan."""
    faiss = pytest.importorskip("faiss")
    source = InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_embedder)
    vectors = np.ascontiguousarray(source.embeddings)
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:6],
        learned_vlad_embedder,
        ExternalSearchIndex.from_faiss_index(flat, name="flat-ip"),
    )
    target = tmp_path_factory.mktemp("rt_fallback") / "store.safetensors"
    written = store.save_to_disk(target)

    with pytest.warns(FutureWarning, match="falling back"):
        loaded = InMemoryImageEmbeddingStore.load_from_disk(written)
    assert isinstance(loaded.index, BruteForceIndex)
    assert loaded.index_name == "brute-force"
    assert loaded.paths == store.paths
