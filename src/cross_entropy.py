"""Cross-entropy loss with optional class weighting.

Implements weighted cross-entropy loss for classification tasks with support for dynamic weight scaling.
"""

import torch

import utils as u


class Cross_Entropy(torch.nn.Module):
    """Cross-entropy loss module with dynamic weight scaling."""

    def __init__(self, args, dataset):
        """Initialize the loss module.

        Args:
            args: Configuration namespace with class_weights.
            dataset: Dataset object for dynamic scaling.
        """
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)

        self.weights = self.dyn_scale(args.task, dataset, weights)

    def dyn_scale(self, task, dataset, weights):
        """Create a dynamic weight scaling function.

        Args:
            task: Task name (link_pred, edge_cls, etc.).
            dataset: Dataset object.
            weights: Base class weights.

        Returns:
            Function to scale weights based on labels.
        """
        # if task == 'link_pred':  commented to have a 1:1 ratio

        #     '''
        #     when doing link prediction there is an extra weighting factor on the non-existing
        #     edges
        #     '''
        #     tot_neg = dataset.num_non_existing
        #     def scale(labels):
        #         cur_neg = (labels == 0).sum(dtype = torch.float)
        #         out = weights.clone()
        #         out[0] *= tot_neg/cur_neg
        #         return out
        # else:
        #     def scale(labels):
        #         return weights
        def scale(labels):
            return weights

        return scale

    def logsumexp(self, logits):
        """Compute log-sum-exp for numerical stability.

        Args:
            logits: Model logits (M x C).

        Returns:
            Log-sum-exp values.
        """
        m, _ = torch.max(logits, dim=1)
        m = m.view(-1, 1)
        sum_exp = torch.sum(torch.exp(logits - m), dim=1, keepdim=True)
        return m + torch.log(sum_exp)

    def forward(self, logits, labels):
        """Compute the weighted cross-entropy loss.

        Args:
            logits: Model predictions (M x C) where M is batch size and C is number of classes.
            labels: Ground truth labels (M,) with integer class IDs.

        Returns:
            Scalar loss value.
        """
        labels = labels.view(-1, 1)
        alpha = self.weights(labels)[labels].view(-1, 1)
        loss = alpha * (-logits.gather(-1, labels) + self.logsumexp(logits))
        return loss.mean()
