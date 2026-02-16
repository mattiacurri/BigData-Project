"""EGCN historical implementation (GRCU with masking and TopK)."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import utils as u


class EGCN(torch.nn.Module):
    """Historical EGCN model with masking-aware GRCU layers."""

    def __init__(self, args, activation, device="cpu", skipfeats=False):
        """Initialize historical EGCN from `args` and activation."""
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node, args.layer_1_feats, args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace(
                {
                    "in_feats": feats[i - 1],
                    "out_feats": feats[i],
                    "activation": activation,
                }
            )

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        """Forward pass that applies masking-aware GRCU layers to the node lists."""
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat(
                (out, node_feats), dim=1
            )  # use node_feats.to_dense() if 2hot encoded input
        return out


class GRCU(torch.nn.Module):
    """Masked GRCU that supports evolving weights conditioned on node embeddings."""

    def __init__(self, args):
        """Initialize masked GRCU with `args`."""
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
        """Initialize tensor `t` in-place using Kaiming uniform initialization."""
        # Use Kaiming initialization for better convergence with ReLU
        torch.nn.init.kaiming_uniform_(t, nonlinearity="relu")

    def forward(self, A_list, node_embs_list, mask_list):
        """Compute GRCU outputs using masks and optionally evolved weights."""
        # Remove batch dimension if present
        node_embs_list = [ne.squeeze(0) if ne.dim() == 3 else ne for ne in node_embs_list]

        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            # Coalesce sparse tensor before matmul to avoid expand issues
            Ahat = Ahat.coalesce()
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    """Matrix GRU cell specialized for historical GRCU (mask-aware)."""

    def __init__(self, args):
        """Initialize mat_GRU_cell gates and helpers from `args`."""
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows, args.cols, torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        """Compute one masked evolution step for the matrix GRU cell."""
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        return (1 - update) * prev_Q + update * h_cap


class mat_GRU_gate(torch.nn.Module):
    """Gate module for mat-GRU: linear transforms plus activation."""

    def __init__(self, rows, cols, activation):
        """Initialize gate parameters for shape (rows, cols)."""
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        """Initialize tensor `t` in-place using Kaiming uniform initialization."""
        # Use Kaiming initialization for better convergence with ReLU
        torch.nn.init.kaiming_uniform_(t, nonlinearity="relu")

    def forward(self, x, hidden):
        """Apply gate linear transforms and activation to `x` with `hidden`."""
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class TopK(torch.nn.Module):
    """Top-k selection helper used by mat-GRU implementations."""

    def __init__(self, feats, k):
        """Initialize TopK with `feats` and `k`."""
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        """Initialize tensor `t` in-place using Kaiming uniform initialization."""
        # Use Kaiming initialization for better convergence with ReLU
        torch.nn.init.kaiming_uniform_(t, nonlinearity="relu")

    def forward(self, node_embs, mask):
        """Select top-k node embeddings using `scorer` and return transposed output."""
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        # Handle sparse tensors
        if isinstance(node_embs, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
            node_embs = node_embs.to_dense()

        # Get number of nodes
        num_nodes = node_embs.size(0)

        # Ensure topk_indices are within bounds (fix for broadcasting issue)
        # When mask is 1D and scores is 2D [num_nodes, 1], broadcasting creates
        # [num_nodes, num_nodes] matrix, so indices can be >= num_nodes
        valid_mask = topk_indices < num_nodes
        topk_indices = topk_indices[valid_mask]

        # If we have no valid indices, create default indices
        if topk_indices.size(0) == 0:
            k_actual = min(self.k, num_nodes)
            topk_indices = torch.arange(k_actual, dtype=torch.long, device=node_embs.device)

        # If we still need more indices, pad with valid indices
        if topk_indices.size(0) < self.k and num_nodes > 0:
            topk_indices = u.pad_with_last_val(topk_indices, min(self.k, num_nodes))
            # Ensure padded indices are also within bounds
            topk_indices = topk_indices % num_nodes

        tanh = torch.nn.Tanh()

        # Index using flattened scores
        scores_flat = scores.view(-1)
        out = node_embs[topk_indices] * tanh(scores_flat[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
