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

### The `-quickgelu` CLIP variants are spellings, not architectures

Whether a CLIP tower uses the QuickGELU activation of the original OpenAI
models or the exact GELU of newer checkpoints is read off the checkpoint
itself, so a variant and its `-quickgelu` twin build the same model for every
tag they share. The plain names exist separately only because some of them
offer extra pretrained tags.
