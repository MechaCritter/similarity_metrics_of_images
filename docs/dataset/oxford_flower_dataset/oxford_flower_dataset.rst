OxfordFlowerDataset
===================

File: ``pyvisim/datasets/datasets.py``

A PyTorch ``Dataset`` for the `Oxford 102 Flowers
<https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ dataset (8189 images
across 102 categories). Indexing yields a ``(image, label, image_path)`` tuple,
where ``image`` is an RGB NumPy array.

.. code-block:: python

   from pyvisim.datasets import OxfordFlowerDataset

   dataset = OxfordFlowerDataset(purpose="train")
   image, label, path = dataset[0]

Iterating yields the same tuple, in the same order:

.. code-block:: python

   import os

   for image, label, path in dataset:
       print("Image shape:", image.shape)
       print("Image label:", label)
       print("Image path:", os.path.basename(path))

What gets downloaded
--------------------

The first instantiation downloads three files from `the website of the
University of Oxford's Visual Geometry Group
<https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html>`_ into the user
cache directory:

- the **dataset images**: 8189 images of 102 flower categories,
- **``imagelabels.mat``**: the category label of every image,
- **``setid.mat``**: the ids assigning each image to the training, validation
  or test split.

The swapped train/test split
----------------------------

The constructor's ``purpose`` accepts ``"train"``, ``"validation"``,
``"test"``, or a list to combine splits (for example
``["train", "validation"]``).

Note the deliberate swap: the original dataset ships 1020 training, 1020
validation and 6149 test images. This class maps the original **test** ids to
``train`` and the original **train** ids to ``test``, so training has the
larger pool. Keep this in mind if you compare results against papers that use
the original split. For more technical information, visit
https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/dataset/arc42.md.

TODO
----

- Implement ``transform`` method

API reference
-------------

.. autoclass:: pyvisim.datasets.OxfordFlowerDataset
   :members:
   :inherited-members: Dataset, Generic
   :show-inheritance:
