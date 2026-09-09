# arc42: distance

Software architecture of `pyvisim.distance`. This document is for developers and
is not part of the published documentation.

## Building block view

The module is a flat set of functions over two 2-D matrices, with no state and
no classes. The embedders reach them by name through their `similarity_func`
argument, which is why the accepted names (`"cosine"`, `"euclidean"`, `"l1"`,
`"manhattan"`) are part of the public surface while the functions behind them
can be reworked freely.

## Architecture decisions

### The result matrix is the memory budget

A pairwise call over `(N, D)` and `(M, D)` inputs is allowed to allocate its
`(N, M)` result and little else. `cosine_similarity` therefore divides the row
norms out of the result in place instead of materializing normalized copies of
either input, so its peak extra memory is the result matrix no matter how large
the inputs are. `manhattan_distances` cannot avoid a broadcast temporary the
same way, so it caps that temporary instead, through the
`working_memory_bytes` keyword.
