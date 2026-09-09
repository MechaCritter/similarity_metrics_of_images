# arc42: dataset

Software architecture of `pyvisim.datasets`. This document is for developers and
is not part of the published documentation.

## Architecture decisions

### The train and test splits are swapped

`OxfordFlowerDataset` maps the original **test** ids to `train` and the original
**train** ids to `test`, turning the shipped 1020/1020/6149 split into a
6149/1020/1020 one. Training therefore has the larger pool, which is what the
clustering models the classic embedders fit benefit from.

The consequence is that numbers measured on this dataset are not comparable to
papers that use the original split, so the swap is stated on the public page as
well.
