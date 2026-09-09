# arc42: neural_networks

Software architecture of `pyvisim.neural_networks`. This document is for
developers and is not part of the published documentation.

## Building block view

Every network in this module is a backbone plus a head. The backbone is the
pretrained convolutional network that turns an image into features, and it is
built by name through `build_backbone`, which also serves the ImageNet
preprocessing the torchvision weights were trained with. `pretrained=False`
returns the bare architecture, which is what deserialization needs: the trained
weights are loaded into it afterwards.

The Siamese and triplet networks realize their several "branches" implicitly by
weight sharing, so there is one backbone instance per model, not one per
branch.

`ClipEmbedder` does not follow that shape. It carries a re-implementation of
the CLIP image tower rather than a torchvision backbone, and it is an embedder
rather than a trainable network in this library.

## Architecture decisions

### A serialized model carries its architecture, not its weights inline

Serialization splits a model into a configuration describing the architecture
(the backbone name, the head sizes, the CLIP variant and pretrained tag) and
the learned weights, which travel separately as the model's `state_dict`. The
consequence is that reconstruction builds the architecture first and loads the
weights into it, so no pretrained checkpoint has to be fetched to restore a
saved model.

### A read-only property must not become an orphan submodule

`torch.nn.Module` overrides `__setattr__` and registers any `torch.nn.Module`
value directly in `self._modules`, so assigning to a read-only property such as
`head` would silently register an orphan submodule instead of failing.
`NeuralImageEmbedder.__setattr__` routes assignments to class-level properties
through `property.__set__`, which restores standard Python semantics.

The same ordering constraint applies to construction: a subclass must call the
`NeuralImageEmbedder` constructor before registering any submodule, so that
`torch.nn.Module` is initialized first.

### Deserialization never downloads a checkpoint

`ClipEmbedder._from_config` builds the tower without its pretrained weights,
because the caller restores them right after from the serialized `state_dict`.
Downloading them first would be both wasteful and a needless network
dependency. The instance is therefore created without running `__init__`, whose
contract is to return a ready-to-use embedder with the pretrained weights
loaded.

### The image preprocessing is serialized as its `repr`

`transforms.Compose` is stateful and hence not JSON-safe, so a serialized
model stores the transform's `repr` rather than the transform. On
deserialization a mismatch against the transform the network was built with is
reported as a warning, and a `transform` that matters has to be passed back in
by the caller.

### The losses are reimplemented on top of torch

Some distance modules are reimplemented with `torch` rather than reused from
`pyvisim.distance`, so that the gradient flows through the loss in the
`forward` pass.

`TripletLoss` mines its triplets online from a labeled batch, exactly as in
FaceNet, and offline triplet selection is deliberately not supported. The
mining strategy sets the memory cost: `"batch_all"` and `"semi_hard"` build a
`(batch, batch, batch)` comparison tensor and so grow cubically with the batch
size, while `"batch_hard"` stays quadratic.

`ContrastiveLoss` caps its margin at 2 because the embeddings are
L2-normalized, under which the maximum distance between two of them is 2:

```
||u - v||^2 = ||u||^2 + ||v||^2 - 2 * (u . v) = 2 * (1 - (u . v))
```

Since `u` and `v` are unit vectors, `u . v` is the cosine similarity, which
ranges from -1 to 1. Plugging in the extremes gives `||u - v||^2` between 0 and
4, so the L2 distance itself ranges from 0 to 2.

### The `-quickgelu` CLIP variants are spellings, not architectures

Whether a CLIP tower uses the QuickGELU activation of the original OpenAI
models or the exact GELU of newer checkpoints is read off the checkpoint
itself, so a variant and its `-quickgelu` twin build the same model for every
tag they share. The plain names exist separately only because some of them
offer extra pretrained tags.
