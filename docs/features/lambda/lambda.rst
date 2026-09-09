Lambda
======

File: ``pyvisim/features/_lambda.py``

``Lambda`` wraps any user-defined function as a feature extractor, so you can
plug a custom descriptor into the embedders without writing a new
``FeatureExtractorBase`` subclass.

Usage
-----

.. code-block:: python

   from pyvisim.features import Lambda

   extractor = Lambda(func=my_descriptor_fn, output_dim=64)

- ``func`` must take a single image (NumPy array), and return a
  ``(N, output_dim)`` array of descriptors.
- ``output_dim`` is supplied explicitly because, unlike SIFT or a CNN layer, an
  arbitrary function has no inspectable descriptor size.

API reference
-------------

.. autoclass:: pyvisim.features.Lambda
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
