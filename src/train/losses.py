"""Loss functions: weighted cross-entropy and focal loss."""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import TrainingConfig

logger = logging.getLogger("medicalxai.losses")


# ---------------------------------------------------------------------------
# Weighted cross-entropy
# ---------------------------------------------------------------------------


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-class frequency weights.

    Args:
        weight: Float tensor of shape ``(num_classes,)``.
        label_smoothing: Smoothing factor in [0, 1).
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=w,
            label_smoothing=self.label_smoothing,
        )


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss for severe class imbalance.

    Reference: Lin et al., 2017 (https://arxiv.org/abs/1708.02002).

    Args:
        gamma: Focusing parameter (γ).  Higher values down-weight easy examples.
        weight: Optional per-class weights (same role as in cross-entropy).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Per-sample, per-class CE
        ce = F.nll_loss(log_p, targets, weight=w, reduction="none")

        # Probability of the true class
        p_t = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Asymmetric BCE Focal loss (multi-label)
# ---------------------------------------------------------------------------


class AsymmetricBCEFocalLoss(nn.Module):
    """Binary cross-entropy focal loss for multi-label classification.

    Combines BCE with logits with per-class positive weights (to handle
    severe label imbalance) and focal modulation to down-weight easy
    negative examples.

    Reference: Ben-Baruch et al., "Asymmetric Loss for Multi-Label
    Classification", ICCV 2021.

    Args:
        pos_weight:  Float tensor of shape ``(num_classes,)``.  Each value is
                     the ratio of negatives to positives for that class.
                     Computed by ``MultiLabelCXRDataset.class_pos_weights()``.
        gamma_neg:   Focusing parameter for negative examples (default 4).
        gamma_pos:   Focusing parameter for positive examples (default 0).
        clip:        Probability clipping floor for negatives to prevent very
                     large gradients (default 0.05).
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  ``(B, C)`` raw model outputs (no sigmoid applied).
            targets: ``(B, C)`` float binary labels in {0, 1}.
        """
        probs = torch.sigmoid(logits)

        # Clip negative probabilities to [0, 1-clip]
        probs_neg = (probs - self.clip).clamp(min=0.0) if self.clip > 0 else probs

        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        xs_pos = probs
        xs_neg = 1.0 - probs_neg

        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs
            pt_neg = probs_neg
            pt = targets * pt_pos + (1.0 - targets) * pt_neg
            one_minus_pt = targets * (1.0 - pt_pos) + (1.0 - targets) * pt_neg
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            focal_w = (one_minus_pt ** gamma).detach()
            loss = loss * focal_w

        # Apply positive weights
        if pw is not None:
            loss = loss * (targets * pw + (1.0 - targets))

        return -loss.mean()


def build_criterion(
    cfg: TrainingConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Return the appropriate loss function based on *cfg*.

    Args:
        cfg: Training configuration block.
        class_weights: For multiclass: float tensor of shape ``(num_classes,)``
            (class-frequency weights).  For multilabel: per-class positive
            weights from ``MultiLabelCXRDataset.class_pos_weights()``.
    """
    strategy = cfg.class_balance_strategy

    gamma = getattr(cfg, "focal_gamma", 2.0)

    if strategy == "multilabel_bce_focal":
        criterion: nn.Module = AsymmetricBCEFocalLoss(
            pos_weight=class_weights,
            gamma_neg=4.0,
            gamma_pos=0.0,
        )
        logger.info(
            "Using AsymmetricBCEFocalLoss (γ_neg=4.0, γ_pos=0.0, pos_weight=%s)",
            class_weights is not None,
        )
    elif strategy == "focal":
        criterion = FocalLoss(gamma=gamma)
        logger.info("Using FocalLoss (γ=%.1f, no class weights)", gamma)
    elif strategy == "class_balanced_focal":
        criterion = FocalLoss(gamma=gamma, weight=class_weights)
        logger.info("Using FocalLoss (γ=%.1f, class-balanced weights=%s)", gamma, class_weights is not None)
    elif strategy in ("weighted_loss", "weighted_sampler"):
        w = class_weights if strategy == "weighted_loss" else None
        criterion = WeightedCrossEntropyLoss(weight=w, label_smoothing=0.1)
        logger.info(
            "Using WeightedCrossEntropyLoss (weighted=%s, smooth=0.1)",
            w is not None,
        )
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")

    return criterion
