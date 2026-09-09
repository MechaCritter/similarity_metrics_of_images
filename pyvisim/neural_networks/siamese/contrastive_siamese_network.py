from ...lazy_import import OptionalImport
from ..backbones import BackboneWithHead

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class ContrastiveSiameseNetwork(BackboneWithHead):
    """
    Siamese network trained with a contrastive loss, proposed in
    `Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping`.

    This network "learns" the similarity metric directly. Two images are passed
    through the same shared-weight ``backbone`` and projection ``head`` to
    produce embeddings, which are L2-normalized so that cosine similarity
    reduces to a dot product. The network is trained so that similar images map
    to nearby embeddings and dissimilar images map far apart. Following diagram
    visualizes this::

        Input Image A ──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embedding A
                        │
                        │ Shared Weights
                        │
        Input Image B ──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embedding B

        Embedding A + Embedding B
                    │
                    ▼
            Contrastive Loss (training) / fixed metric, e.g. cosine (inference)

    `Contrastive loss` is used to train this network, which has the formula:

    .. math::
        L = \\frac{1}{2N} \\sum_{i=1}^{N} \\Bigl( y_i \\, D_i^2 + (1 - y_i) \\, \\max(0, m - D_i)^2 \\Bigr)

    References:
    ===========
    [1] Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping. In Proceedings of the 2006 IEEE
    Computer Society Conference on Computer Vision and Pattern Recognition
    (CVPR), Vol. 2, 1735-1742. https://doi.org/10.1109/CVPR.2006.100

    :param backbone: name of feature-extraction network. See
        ``https://mechacritter.github.io/Python-Visual-Similarity/neural_networks/backbones/backbones.html``.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param similarity_func: Name of the built-in similarity metric used to score
        two embeddings. One of ``"cosine"``, ``"euclidean"``, ``"l1"``
        or ``"manhattan"``.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``ContrastiveSiameseNetwork`` from a
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

        During training, call this once per branch of a pair and feed both
        embedding batches to the contrastive loss.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: L2-normalized embeddings of shape (batch, embedding_dim).
        """
        return self._forward_once(x)
