"""EGCN model components: GRCU, mat-GRU cells and TopK helper."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import utils as u


class EGCN(torch.nn.Module):
    """Stacked GRCU-based encoder producing node embeddings for each snapshot."""

    def __init__(self, args, activation, device="cpu", skipfeats=False):
        """Initialize EGCN layers from `args` and provided activation."""
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
        """Compute forward pass over a sequence of adjacency matrices and node lists."""
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
    """Graph recurrent convolutional unit that evolves weights over time."""

    def __init__(self, args):
        """Initialize GRCU with `args` (in/out features and activation)."""
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

    def forward(self, A_list, node_embs_list):  # ,mask_list):
        """Apply evolving-GCN weights over `A_list` and `node_embs_list` to compute outputs."""
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
            # Coalesce sparse tensor before matmul to avoid expand issues
            Ahat = Ahat.coalesce()
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    """Matrix GRU cell used to evolve GCN weight matrices over time."""

    def __init__(self, args):
        """Initialize mat_GRU_cell gates and helpers from `args`."""
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows, args.cols, torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q):  # ,prev_Z,mask):
        """Compute one evolution step for the matrix GRU cell."""
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        return (1 - update) * prev_Q + update * h_cap


class mat_GRU_gate(torch.nn.Module):
    """Gate module used by matrix GRU cells (linear transforms + activation)."""

    def __init__(self, rows, cols, activation):
        """Initialize linear parameters for the gate."""
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
        """Apply gate transformation to input `x` with `hidden` state."""
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class TopK(torch.nn.Module):
    """Top-k selector used to pick strongest node embedding entries."""

    def __init__(self, feats, k):
        """Initialize TopK selector with `feats` and `k`."""
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

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
