"""GCN-based model architectures.

Contains various sparse GCN variants including LSTM/GRU temporal models and classifiers.
"""

import torch
from torch.nn import Linear, ReLU, Sequential


class Classifier(torch.nn.Module):
    """Classification head for node pair or edge classification."""

    def __init__(self, args, out_features=2, in_features=None):
        """Initialize the classifier.

        Args:
            args: Configuration namespace with layer dimensions.
            out_features: Number of output classes (default: 2 for link prediction).
            in_features: Number of input features. If None, computed from args.
        """
        super(Classifier, self).__init__()

        print(f"in_features for classifier: {in_features}")

        self.mlp = Sequential(
            Linear(in_features=in_features, out_features=args.gcn_parameters["cls_feats"]),
            ReLU(),
            Linear(in_features=args.gcn_parameters["cls_feats"], out_features=out_features),
        )

        # print number of parameters
        total_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"MLP Classifier initialized with {total_params} trainable parameters.")

    def forward(self, x):
        """Forward pass through classifier MLP.

        Args:
            x: Input node pair embeddings.

        Returns:
            Classification logits for node pair relationship.
        """
        return self.mlp(x)
