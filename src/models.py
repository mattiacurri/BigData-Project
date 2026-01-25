"""GCN-based model architectures.

Contains various sparse GCN variants including LSTM/GRU temporal models and classifiers.
"""

import torch
from torch.nn import GRU, LSTM, Linear, Module, ParameterList, ReLU, Sequential
from torch.nn.parameter import Parameter

import utils as u


class Sp_GCN(Module):
    """Sparse Graph Convolutional Network."""

    def __init__(self, args, activation):
        """Initialize the GCN.

        Args:
            args: Configuration namespace with layer dimensions.
            activation: Activation function to use.
        """
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = ParameterList()
        for i in range(self.num_layers):
            if i == 0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                u.reset_param(w_i)
            self.w_list.append(w_i)

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        """Forward pass of GCN.

        Args:
            A_list: List of sparse adjacency matrices for each time step.
            Nodes_list: List of node feature matrices.
            nodes_mask_list: List of node masks.

        Returns:
            Node embeddings from final GCN layer.
        """
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l


class Sp_Skip_GCN(Sp_GCN):
    """Sparse GCN with skip connections."""

    def __init__(self, args, activation):
        """Initialize skip-connection GCN.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self, A_list, Nodes_list=None):
        """Forward pass with skip connections.

        Args:
            A_list: List of sparse adjacency matrices.
            Nodes_list: List of node feature matrices.

        Returns:
            Node embeddings with skip connections.
        """
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2


class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    """Sparse GCN with skip connections that preserve node features."""

    def __init__(self, args, activation):
        """Initialize GCN with node feature skip connections.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)

    def forward(self, A_list, Nodes_list=None):
        """Forward pass with node feature skip connections.

        Args:
            A_list: List of sparse adjacency matrices.
            Nodes_list: List of node feature matrices.

        Returns:
            Concatenated output with original node features.
        """
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat(
            (last_l, node_feats), dim=1
        )  # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l


class Sp_GCN_LSTM_A(Sp_GCN):
    """GCN followed by LSTM for temporal sequence modeling (Type A)."""

    def __init__(self, args, activation):
        """Initialize GCN-LSTM model.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)
        self.rnn = LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        """Forward pass through temporal graph convolutions and LSTM.

        Args:
            A_list: List of sparse adjacency matrices across time steps.
            Nodes_list: List of node feature matrices across time steps.
            nodes_mask_list: Optional mask for nodes.

        Returns:
            Final LSTM hidden state representing temporal graph embeddings.
        """
        last_l_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            # A_list: T, each element sparse tensor
            # note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    """GCN followed by GRU for temporal sequence modeling (Type A)."""

    def __init__(self, args, activation):
        """Initialize GCN-GRU model.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)
        self.rnn = GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )


class Sp_GCN_LSTM_B(Sp_GCN):
    """GCN with LSTM applied at layer level (Type B)."""

    def __init__(self, args, activation):
        """Initialize GCN-LSTM type B model.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)
        assert args.num_layers == 2, "GCN-LSTM and GCN-GRU requires 2 conv layers."
        self.rnn_l1 = LSTM(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers,
        )

        self.rnn_l2 = LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        """Forward pass through GCN layers and dual LSTMs at different depths.

        Args:
            A_list: List of sparse adjacency matrices across time steps.
            Nodes_list: List of node feature matrices across time steps.
            nodes_mask_list: Optional mask for nodes.

        Returns:
            Final LSTM hidden state from layer 2 representing temporal embeddings.
        """
        l1_seq = []
        l2_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            # A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    """GCN with GRU applied at layer level (Type B)."""

    def __init__(self, args, activation):
        """Initialize GCN-GRU type B model.

        Args:
            args: Configuration namespace.
            activation: Activation function.
        """
        super().__init__(args, activation)
        self.rnn_l1 = GRU(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers,
        )

        self.rnn_l2 = GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )


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

        # if in_features is not None:
        # num_feats = in_features
        # elif args.experiment_type in [
        #     "sp_lstm_A_trainer",
        #     "sp_lstm_B_trainer",
        #     "sp_weighted_lstm_A",
        #     "sp_weighted_lstm_B",
        # ]:
        #     num_feats = args.gcn_parameters["lstm_l2_feats"] * 2
        # else:
        #     num_feats = args.gcn_parameters["layer_2_feats"] * 2
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
