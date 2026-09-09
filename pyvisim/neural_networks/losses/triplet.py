"""Implements the Triplet Loss."""

import functools
from collections.abc import Callable
from typing import Any

from ...lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch

_MINING_STRATEGIES = ("batch_all", "batch_hard", "semi_hard")
_ForwardMethod = Callable[[Any, "torch.Tensor", "torch.Tensor"], "torch.Tensor"]


def _pairwise_euclidean_distances(
    embeddings: torch.Tensor, squared: bool = False
) -> torch.Tensor:
    # Uses the dot-product expansion ``||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2``.
    dot_products = embeddings @ embeddings.T
    squared_norms = dot_products.diagonal()
    distances_sq = (
        squared_norms.unsqueeze(0) - 2 * dot_products + squared_norms.unsqueeze(1)
    )
    distances_sq = distances_sq.clamp(min=0.0)
    if squared:
        return distances_sq
    zero_mask = distances_sq == 0

    # Gradient of sqrt is infinite at 0 => small epsilon added to avoid NaN
    distances = torch.sqrt(distances_sq + zero_mask * 1e-16)
    return distances * ~zero_mask


def _validate_embeddings_and_labels(forward: _ForwardMethod) -> _ForwardMethod:
    @functools.wraps(forward)
    def wrapper(
        self: Any, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must be 2-dimensional (batch, dim), "
                f"got shape {tuple(embeddings.shape)}."
            )
        if labels.dim() != 1:
            raise ValueError(
                f"labels must be 1-dimensional, got shape {tuple(labels.shape)}."
            )
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {labels.shape[0]} labels for "
                f"{embeddings.shape[0]} embeddings."
            )
        return forward(self, embeddings, labels)

    return wrapper


def _positive_pair_mask(labels: torch.Tensor) -> torch.Tensor:
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_pair = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    return same_label & ~self_pair


def _negative_pair_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.unsqueeze(0) != labels.unsqueeze(1)


def _zero_loss(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings.sum() * 0.0


class TripletLoss(torch.nn.Module):
    """
    Triplet loss with online triplet mining, proposed in
    `Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified
    Embedding for Face Recognition and Clustering`.

    For a triplet of anchor ``a``, positive ``p`` (same class) and negative
    ``n`` (different class), the loss is the hinge

    ``L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)``

    where ``d`` is the (optionally squared) Euclidean distance. Instead of
    receiving precomputed triplets, the loss mines them *online* from a
    labeled batch of embeddings, exactly as in FaceNet: every image in the
    batch acts as an anchor and its partners are picked from the same batch.
    Offline triplet selection is deliberately not supported.

    Supported mining strategies:

    - ``"semi_hard"`` (**default**, FaceNet): for every positive pair
      ``(a, p)``, pick the closest negative that is still farther than the
      positive (``d(a, p) < d(a, n)``). If no such negative exists in the
      batch, fall back to the farthest negative of the anchor. The loss is
      averaged over all positive pairs.
    - ``"batch_hard"``: for every anchor, use only its farthest positive and
      its closest negative (Hermans et al., 2017). The loss is averaged over
      all anchors that have at least one positive and one negative.
    - ``"batch_all"``: use every valid triplet in the batch and average over
      the triplets that violate the margin. Averaging over *all* triplets
      instead would let the many trivially satisfied ones wash out the signal
      (Hermans et al., 2017).

    NOTE
    ----
    ``"batch_all"`` and ``"semi_hard"`` build a ``(batch, batch, batch)``
    comparison tensor, so their memory cost grows cubically with the batch
    size; ``"batch_hard"`` stays quadratic.

    A batch that yields no valid triplet (e.g. it contains only one class)
    produces a zero loss that is still connected to the autograd graph, so
    ``loss.backward()`` keeps working in the training loop.

    References:
    ===========
    [1] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A
    Unified Embedding for Face Recognition and Clustering. CVPR.
    https://doi.org/10.1109/CVPR.2015.7298682

    [2] Hoffer, E., & Ailon, N. (2014). Deep Metric Learning Using Triplet
    Network. https://arxiv.org/abs/1412.6622

    [3] Hermans, A., Beyer, L., & Leibe, B. (2017). In Defense of the Triplet
    Loss for Person Re-Identification. https://arxiv.org/abs/1703.07737

    :param margin: Margin enforced between positive and negative distances.
        The default ``0.2`` is FaceNet's setting for squared distances.
    :param mining: Online mining strategy, one of ``"semi_hard"`` (default),
        ``"batch_hard"`` or ``"batch_all"``.
    :param squared: If ``True`` (default), use squared Euclidean distances as
        in FaceNet. Hermans et al. report better convergence with plain
        distances (``False``), typically combined with ``"batch_hard"``.
    :raises ValueError: If ``margin`` is not strictly positive or ``mining``
        is not a supported strategy.
    """

    def __init__(
        self,
        margin: float = 0.2,
        mining: str = "semi_hard",
        squared: bool = True,
    ) -> None:
        _torch_import.check()
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}.")
        if mining not in _MINING_STRATEGIES:
            raise ValueError(
                f"Unsupported mining strategy: {mining!r}. "
                f"Supported strategies: {', '.join(map(repr, _MINING_STRATEGIES))}."
            )
        self.margin = margin
        self.mining = mining
        self.squared = squared

    @_validate_embeddings_and_labels
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the mined triplet loss over a labeled embedding batch.

        :param embeddings: Embeddings of shape (batch, dim).
        :param labels: Class labels of shape (batch,); any dtype supporting
            equality comparison (integers in practice).
        :return: The scalar loss.
        :raises ValueError: If ``embeddings`` is not 2-dimensional, if ``labels``
            is not 1-dimensional, or if the two disagree on the batch size.
        """
        distances = _pairwise_euclidean_distances(embeddings, squared=self.squared)
        pos_mask = _positive_pair_mask(labels)
        neg_mask = _negative_pair_mask(labels)

        if self.mining == "batch_all":
            return self._batch_all(embeddings, distances, pos_mask, neg_mask)
        if self.mining == "batch_hard":
            return self._batch_hard(embeddings, distances, pos_mask, neg_mask)
        return self._semi_hard(embeddings, distances, pos_mask, neg_mask)

    def _batch_all(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Averages the hinge over every margin-violating triplet in the batch.

        :param embeddings: Embeddings of shape (batch, dim).
        :param distances: Pairwise distance matrix of shape (batch, batch).
        :param pos_mask: Positive-pair mask of shape (batch, batch).
        :param neg_mask: Negative-pair mask of shape (batch, batch).
        :return: The scalar loss.
        """
        # hinge[a, p, n] = d(a, p) - d(a, n) + margin
        hinge = torch.relu(
            distances.unsqueeze(2) - distances.unsqueeze(1) + self.margin
        )
        valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        losses = hinge[valid]
        active = losses[losses > 0]
        if active.numel() == 0:
            return _zero_loss(embeddings)
        return active.mean()

    def _batch_hard(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Averages the hinge of the hardest triplet of each anchor.

        :param embeddings: Embeddings of shape (batch, dim).
        :param distances: Pairwise distance matrix of shape (batch, batch).
        :param pos_mask: Positive-pair mask of shape (batch, batch).
        :param neg_mask: Negative-pair mask of shape (batch, batch).
        :return: The scalar loss.
        """
        valid_anchor = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not bool(valid_anchor.any()):
            return _zero_loss(embeddings)
        inf = float("inf")
        hardest_pos = (
            distances.masked_fill(~pos_mask, -inf)[valid_anchor].max(dim=1).values
        )
        hardest_neg = (
            distances.masked_fill(~neg_mask, inf)[valid_anchor].min(dim=1).values
        )
        return torch.relu(hardest_pos - hardest_neg + self.margin).mean()

    def _semi_hard(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Averages the hinge over positive pairs with semi-hard negatives.

        For each positive pair the closest negative farther than the positive
        is selected; pairs without one fall back to the anchor's farthest
        negative, mirroring the reference TensorFlow implementation of
        FaceNet's strategy.

        :param embeddings: Embeddings of shape (batch, dim).
        :param distances: Pairwise distance matrix of shape (batch, batch).
        :param pos_mask: Positive-pair mask of shape (batch, batch).
        :param neg_mask: Negative-pair mask of shape (batch, batch).
        :return: The scalar loss.
        """
        has_neg = neg_mask.any(dim=1)
        valid_pair = pos_mask & has_neg.unsqueeze(1)
        if not bool(valid_pair.any()):
            return _zero_loss(embeddings)

        inf = float("inf")
        batch_size = distances.shape[0]
        # semi_hard[a, p, n]: n is a negative of a and farther away than p.
        d_an = distances.unsqueeze(1).expand(batch_size, batch_size, batch_size)
        semi_hard = neg_mask.unsqueeze(1) & (d_an > distances.unsqueeze(2))
        closest_semi_hard = d_an.masked_fill(~semi_hard, inf).min(dim=2).values
        farthest_neg = distances.masked_fill(~neg_mask, -inf).max(dim=1).values
        selected_neg = torch.where(
            semi_hard.any(dim=2), closest_semi_hard, farthest_neg.unsqueeze(1)
        )
        hinge = torch.relu(distances - selected_neg + self.margin)
        return hinge[valid_pair].mean()
