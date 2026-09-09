# arc42: pixelwise

Software architecture of `pyvisim.pixelwise`. This document is for developers
and is not part of the published documentation.

## Building block view

`PSNR` is a dense metric: it compares two images directly instead of going
through an intermediate vector embedding. It derives from `DenseMetricBase`,
which owns the shared pipeline around the metric itself, namely input
normalization, shape validation and memory-bounded pair batching.

The squared differences are summed by a compiled OpenMP kernel under
`pyvisim/pixelwise/_kernel/`, which `make build-ext` regenerates from its
Cython source. This is also why every compared pair must share the same
`(H, W[, C])` shape: the kernel walks the two buffers in step.

## Architecture decisions

### The kernel is multithreaded and its team size is an environment variable

`PYVISIM_NUM_THREADS` sets the OpenMP team size of the compiled kernels, 4 by
default, and it is read on every kernel call rather than cached, so it can be
changed at runtime through `os.environ`. `batch_size` bounds how many image
pairs enter one kernel call, which is what caps the peak memory of a large
gallery.
