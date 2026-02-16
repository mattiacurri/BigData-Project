"""Model heads and simple classifiers used in experiments."""

import torch


class Classifier(torch.nn.Module):
    """Simple MLP classifier head used after GCN embeddings."""

    def __init__(self, args, out_features=2, in_features=None):
        """Construct classifier MLP using parameters from `args`."""
        super(Classifier, self).__init__()
        activation = torch.nn.ReLU()

        if in_features is not None:
            num_feats = in_features
        else:
            num_feats = args.gcn_parameters["layer_2_feats"] * 2

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_feats, out_features=args.gcn_parameters["cls_feats"]),
            activation,
            torch.nn.Linear(
                in_features=args.gcn_parameters["cls_feats"], out_features=out_features
            ),
        )

        # Initialize weights with Kaiming and biases to zero
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Weight init hook used by `self.apply` to initialize Linear layers."""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the classifier MLP."""
        return self.mlp(x)
