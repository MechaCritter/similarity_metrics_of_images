from typing import cast

from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput
from ..backbones import BackboneWithHead

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class BCESiameseNetwork(BackboneWithHead):
    """
    Siamese network that classifies image pairs, proposed in
    `Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition`.

    Both images are passed through the same shared-weight ``backbone`` and
    projection ``head``; each branch output is squashed with a sigmoid into a
    feature vector ``h in (0, 1)^D`` (the paper's final fully-connected layer
    uses sigmoid units). The two branches are then combined by their
    component-wise L1 distance, and a single learned linear layer maps that
    distance vector to the probability of the pair showing the same class:

    .. math::

        p(x_1, x_2)
        = \\sigma\\Bigl(\\sum_{j} \\alpha_j \\, \\bigl| h_{1,j} - h_{2,j} \\bigr|
        + b\\Bigr)

    where the weights :math:`\\alpha_j` learn the importance of each feature
    dimension, so unlike :class:`ContrastiveSiameseNetwork` the comparison
    metric itself is trained. The network is a binary classifier over pairs and
    is trained with binary cross-entropy on labels ``1`` (same class) / ``0``
    (different class); :meth:`forward` returns raw logits so it composes with
    :class:`torch.nn.BCEWithLogitsLoss` in a numerically stable way.

    Following diagram visualizes this::

        Input Image A ──► Backbone ──► Embedding Head ──► Sigmoid ──► Features A ─┐
                        │                                                   ├─► |A - B| ──► Scoring Layer ──► P(same class)
        Input Image B ──► Backbone ──► Embedding Head ──► Sigmoid ──► Features B ─┘
                (Shared Weights)

    NOTE
    ----
    The score is a *learned probability*, not a geometric similarity: it is
    symmetric in its inputs (the L1 distance is), lives in ``(0, 1)``, and for
    two identical images equals ``sigmoid(b)`` -- the learned bias sets the
    operating point, so a perfect match does not score exactly ``1``.

    References:
    ===========
    [1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.
    https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

    :param backbone: name of feature-extraction network. Default: ``"resnet18"``.
        See
        ``https://mechacritter.github.io/Python-Visual-Similarity/docs/sphinx/_build/html/neural_networks/backbones.html``.
    :param embedding_dim: Dimensionality of the twin feature vectors that the
        scoring layer compares.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``BCESiameseNetwork`` from a
        checkpoint, set this to ``False`` to avoid downloading the weights again.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``embedding_dim`` is not a positive integer or if
        ``backbone`` is not a supported backbone name.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
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
            batch_size=batch_size,
        )
        self._scorer: torch.nn.Module = torch.nn.Linear(embedding_dim, 1)
        self.to(torch.device(device))

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes sigmoid-activated feature vectors for a batch of images.

        Unlike the contrastive variant, the features are *not* L2-normalized;
        each component is squashed into ``(0, 1)`` so the component-wise L1
        distances fed to the scoring layer are bounded.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: Feature tensor of shape (batch, embedding_dim) with values
            in ``(0, 1)``.
        """
        features = self._backbone(x)
        return torch.sigmoid(self._head(features))

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """Not implemented for this class. Please do not use!"""
        raise NotImplementedError(
            f"{type(self).__name__} does not learn to generate embeddings. "
            "Use the ContrastiveSiameseNetwork for that purpose."
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes same-class logits for a batch of aligned image pairs.

        The i-th logit scores the pair ``(x1[i], x2[i])``; apply
        :func:`torch.sigmoid` to obtain probabilities, or feed the logits
        directly to :class:`torch.nn.BCEWithLogitsLoss` during training.

        :param x1: First preprocessed image batch, shape (batch, channels, H, W).
        :param x2: Second preprocessed image batch of the same shape.
        :return: Logit tensor of shape (batch,).
        :raises ValueError: If the two batches differ in shape.
        """
        if x1.shape != x2.shape:
            raise ValueError(
                f"Input batches must have the same shape, got "
                f"{tuple(x1.shape)} vs {tuple(x2.shape)}."
            )
        features1 = self._forward_once(x1)
        features2 = self._forward_once(x2)
        return self._score_distances(torch.abs(features1 - features2))

    def _score_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Maps component-wise L1 distance vectors to same-class logits.

        :param distances: Tensor of shape (..., embedding_dim) holding
            ``|h_1 - h_2|`` for each pair.
        :return: Logit tensor of shape (...,).
        """
        return cast(torch.Tensor, self._scorer(distances).squeeze(-1))

    @torch.no_grad()
    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        features1 = self._embed_images(images1, dims=dims, value_range=value_range)
        features2 = self._embed_images(images2, dims=dims, value_range=value_range)
        distances = torch.abs(features1.unsqueeze(1) - features2.unsqueeze(0))
        probabilities = torch.sigmoid(self._score_distances(distances))
        return cast(FloatNumpyArray, probabilities.cpu().numpy())

    @property
    def scorer(self) -> torch.nn.Module:
        """The learned layer mapping L1 distances to same-class logits."""
        return self._scorer
