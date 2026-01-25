"""EGCN-O (Evolving Graph Convolutional Network - Output variant) model definition.

Implements evolving graph convolutions with simplified weight evolution for temporal graphs.

Modified from: https://github.com/IBM/EvolveGCN/blob/master/egcn_h.py
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import utils as u


class EGCN(torch.nn.Module):
    """Evolving Graph Convolutional Network (Output variant)."""

    def __init__(self, args, activation, device="cpu", skipfeats=False):
        """Initialize EGCN-O model.

        Args:
            args: Configuration namespace with layer dimensions.
            activation: Activation function to use.
            device: Device to place model on.
            skipfeats: Whether to skip connection original features.
        """
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node, args.layer_1_feats, args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace(
                {"in_feats": feats[i - 1], "out_feats": feats[i], "activation": activation}
            )

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        """Forward pass through EGCN-O.

        Args:
            A_list: List of sparse adjacency matrices.
            Nodes_list: List of node feature matrices.
            nodes_mask_list: List of node masks (unused in EGCN-O).

        Returns:
            Final node embeddings.
        """
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)  # ,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat(
                (out, node_feats), dim=1
            )  # use node_feats.to_dense() if 2hot encoded input
        return out


class GRCU(torch.nn.Module):
    """Graph Recurrent Convolution Unit (Output variant)."""

    def __init__(self, args):
        """Initialize GRCU.

        Args:
            args: Configuration with input/output feature dimensions.
        """
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        """Initialize tensor with uniform distribution.

        Args:
            t: Tensor to initialize.
        """
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list):
        """Forward pass of GRCU.

        Args:
            A_list: List of adjacency matrices.
            node_embs_list: List of node embeddings for each time step.

        Returns:
            List of updated node embeddings across time.
        """
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    """Matrix-based GRU cell for weight evolution (simplified)."""

    def __init__(self, args):
        """Initialize matrix GRU cell.

        Args:
            args: Configuration with row/column dimensions.
        """
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows, args.cols, torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q):  # ,prev_Z,mask):
        """Forward pass of matrix GRU cell.

        Args:
            prev_Q: Previous weight matrix.

        Returns:
            Updated weight matrix.
        """
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    """GRU gate for matrix operations."""

    def __init__(self, rows, cols, activation):
        """Initialize GRU gate.

        Args:
            rows: Number of rows in weight matrices.
            cols: Number of columns.
            activation: Activation function.
        """
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        """Initialize tensor with uniform distribution.

        Args:
            t: Tensor to initialize.
        """
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        """Forward pass of GRU gate.

        Args:
            x: Input feature matrix.
            hidden: Hidden state.

        Returns:
            Gate output.
        """
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)

        return out


class TopK(torch.nn.Module):
    """Top-K node selector with learnable scores."""

    def __init__(self, feats, k):
        """Initialize TopK selector.

        Args:
            feats: Number of features (input dimension).
            k: Number of top nodes to select.
        """
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        """Initialize tensor with uniform distribution.

        Args:
            t: Tensor to initialize.
        """
        # Initialize based on the number of rows
        stdv = 1.0 / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        """Forward pass to select top-K nodes.

        Args:
            node_embs: Node embeddings.
            mask: Node mask for masking.

        Returns:
            Weighted embeddings of top-K nodes.
        """
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.SparseTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
