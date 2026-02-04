"""Custom loss functions for temporal graph learning.

Includes losses specifically designed for imbalanced classification tasks
common in link prediction and anomaly detection on social networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    Focal Loss down-weights easy examples and focuses on hard examples,
    making it effective for imbalanced datasets where the minority class
    (e.g., risky links) is rare but important.

    Reference: Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Weighting factor for the rare class (default: 1.0).
               Can be a scalar or tensor of shape [num_classes].
        gamma: Focusing parameter (default: 2.0). Higher values focus more on hard examples.
        weight: Optional class weights tensor for additional balancing.
        reduction: Reduction method ('mean', 'sum', 'none').

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction="mean"):
        """Initialize FocalLoss.

        Args:
            alpha: Weighting factor for the rare class (default: 1.0).
                   Can be a scalar or tensor of shape [num_classes].
            gamma: Focusing parameter (default: 2.0). Higher values focus more on hard examples.
            weight: Optional class weights tensor for additional balancing.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss.

        Args:
            inputs: Logits tensor of shape [N, C] where C is number of classes.
            targets: Ground truth class indices of shape [N].

        Returns:
            Scalar loss value (or tensor if reduction='none').
        """
        # Compute cross entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")

        # Get predicted probabilities for the target class
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE) = softmax probability of correct class

        # Apply focal weighting: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            # Alpha is a list of weights for each class
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
        else:
            # Scalar alpha (typically for binary: alpha for positive class)
            alpha_t = self.alpha

        # Compute focal loss
        loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BCEFocalLoss(nn.Module):
    """Focal Loss for binary classification with BCE.

    Alternative implementation for binary classification tasks using
    BCEWithLogitsLoss as the base. Useful when you want single-output
    binary classification instead of 2-class softmax.

    Args:
        alpha: Weighting factor for positive class (default: 0.25).
        gamma: Focusing parameter (default: 2.0).
        pos_weight: Optional weight for positive class samples.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction="mean"):
        """Initialize BCEFocalLoss.

        Args:
            alpha: Weighting factor for positive class (default: 0.25).
            gamma: Focusing parameter (default: 2.0).
            pos_weight: Optional weight for positive class samples.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss for binary classification.

        Args:
            inputs: Logits tensor of shape [N, 1] or [N].
            targets: Binary targets of shape [N] with values {0, 1}.

        Returns:
            Scalar loss value.
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )

        # Get predicted probabilities
        pt = torch.exp(-bce_loss)  # p_t for BCE

        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting (alpha for positive class, 1-alpha for negative)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1 - self.alpha, device=inputs.device),
        )

        # Compute focal loss
        loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for classification tasks.

    Particularly effective for imbalanced datasets. Similar to F1 score optimization.

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0).
        reduction: Reduction method ('mean', 'sum').
    """

    def __init__(self, smooth=1.0, reduction="mean"):
        """Initialize DiceLoss.

        Args:
            smooth: Smoothing factor to avoid division by zero (default: 1.0).
            reduction: Reduction method ('mean', 'sum').
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute dice loss.

        Args:
            inputs: Logits tensor of shape [N, C].
            targets: Class indices of shape [N].

        Returns:
            Scalar loss value.
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)

        # Convert targets to one-hot
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute dice coefficient for each class
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - dice
        loss = 1 - dice

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss with weighting for FP/FN.

    Allows controlling the trade-off between false positives and false negatives.

    Args:
        alpha: Weight for false negatives (default: 0.5).
        beta: Weight for false positives (default: 0.5).
        smooth: Smoothing factor (default: 1.0).
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """Initialize TverskyLoss.

        Args:
            alpha: Weight for false negatives (default: 0.5).
            beta: Weight for false positives (default: 0.5).
            smooth: Smoothing factor (default: 1.0).
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """Compute Tversky loss."""
        probs = F.softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # True positives, false positives, false negatives
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        return (1 - tversky).mean()
