from ...lazy_import import OptionalImport
from ..backbones import BackboneWithHead

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class TripletNeuralNetwork(BackboneWithHead):
    """
    Triplet network for image similarity, proposed in
    `Hoffer, E., & Ailon, N. (2014). Deep Metric Learning Using Triplet
    Network` and popularized by `Schroff, F., Kalenichenko, D., & Philbin, J.
    (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering`.

    A triplet network is a shared-weight embedding network trained on
    triplets of anchor, positive (same class) and negative (different class)
    images: the anchor is pulled towards the positive and pushed away from
    the negative by at least a margin. The three classic "branches" of the
    architecture are realized implicitly by weight sharing: every image is
    passed through the same ``backbone`` and projection ``head``, and the
    embeddings are L2-normalized so that cosine similarity reduces to a dot
    product. Following diagram visualizes this::

        Anchor   ──┐
        Positive ──┼──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embeddings
        Negative ──┘     (Shared Weights)

        Embedding A + Embedding P + Embedding N
                            │
                            ▼
            Triplet Loss (training) / fixed metric, e.g. cosine (inference)

    `Triplet loss` is used to train this network, which has the formula:

    .. math::
        L(a, p, n) = \\max\\bigl(0, \\, d(a, p) - d(a, n) + m\\bigr)

    Training follows FaceNet's *online mining* scheme exclusively: instead of
    preparing (anchor, positive, negative) files offline, a labeled batch of
    images is passed through :meth:`forward` once and
    :class:`pyvisim.neural_networks.losses.TripletLoss` mines the triplets
    from the batch itself.

    References:
    ===========
    [1] Hoffer, E., & Ailon, N. (2014). Deep Metric Learning Using Triplet
    Network. https://arxiv.org/abs/1412.6622

    [2] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A
    Unified Embedding for Face Recognition and Clustering. CVPR.
    https://doi.org/10.1109/CVPR.2015.7298682

    :param backbone: name of feature-extraction network. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/docs/sphinx/_build/html/neural_networks/backbones.html``.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param similarity_func: Name of the built-in similarity metric used to score
        two embeddings. One of ``"cosine"``, ``"euclidean"``, ``"l1"``
        or ``"manhattan"``.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``TripletNeuralNetwork`` from a
        checkpoint, set this to ``False`` to avoid downloading the weights again.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``embedding_dim`` is not a positive integer, if
        ``backbone`` is not a supported backbone name, or if ``similarity_func``
        is not a supported similarity metric.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        similarity_func: str = "cosine",
        transform: transforms.Compose | None = None,
        device: str | torch.device = "cpu",
        pretrained_backbone: bool = True,
        *,
        batch_size: int = 16,
    ):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            transform=transform,
            pretrained_backbone=pretrained_backbone,
            similarity_func=similarity_func,
            batch_size=batch_size,
        )
        self.to(torch.device(device))

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for a batch of preprocessed images.

        The embeddings are unit-length, so cosine similarity between two of them
        equals their dot product.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: L2-normalized embeddings of shape (batch, embedding_dim).
        """
        features = self._backbone(x)
        embeddings = self._head(features)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for a batch of preprocessed images.

        During training this single shared-weight pass replaces the three
        explicit triplet branches: feed a labeled batch through it and mine the
        triplets online with
        :class:`pyvisim.neural_networks.losses.TripletLoss`.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: L2-normalized embeddings of shape (batch, embedding_dim).
        """
        return self._forward_once(x)
