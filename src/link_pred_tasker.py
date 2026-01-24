"""Link prediction tasker for temporal graph learning.

Handles the preparation and sampling of data for link prediction tasks on temporal graphs.
"""

import torch

import taskers_utils as tu
import utils as u


class Link_Pred_Tasker:
    """Creates a tasker object which computes the required inputs for training on a link prediction task.

    It receives a dataset object which should have two attributes: nodes_feats and edges, this
    makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
    structure).

    Based on the dataset it implements the get_sample function required by edge_cls_trainer.
    This is a dictionary with:
            - time_step: the time_step of the prediction
            - hist_adj_list: the input adjacency matrices until t, each element of the list
                                             is a sparse tensor with the current edges.
            - nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
                                              two dimmensions: node_idx and node_feats
            - label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2
                                     matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
                                     , 0 if it doesn't
    """

    def __init__(self, args, dataset):
        """Initialize the link prediction tasker.

        Args:
                args: Configuration namespace.
                dataset: Dataset object with edges and node features.
        """
        self.data = dataset
        # max_time for link pred should be one before
        self.max_time = dataset.max_time - 1
        self.args = args
        self.num_classes = 2

        # Use dataset features (BERT embeddings with temporal support)
        self.feats_per_node = dataset.feats_per_node
        print(f"Using temporal node features: {self.feats_per_node} features per node")

        self.get_node_feats = self.build_get_node_feats(args, dataset)
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)

    def build_prepare_node_feats(self, args, dataset):
        """Build the node feature preparation function.

        Args:
                args: Configuration namespace.
                dataset: Dataset object.

        Returns:
                Function to prepare node features.
        """
        return self.data.prepare_node_feats

    def build_get_node_feats(self, args, dataset):
        """Build the node feature extraction function.

        Args:
                args: Configuration namespace specifying feature type.
                dataset: Dataset object.

        Returns:
                Function to extract temporal node features.
        """
        print("Using temporal node features from dataset (BERT embeddings per snapshot)")

        def get_node_feats(adj, timestep):
            """Get temporal node features up to the given timestep.

            Args:
                adj: Adjacency matrix (unused but kept for API compatibility).
                timestep: Current timestep to compute features for.

            Returns:
                Node features tensor computed from posts up to this timestep.
            """
            return dataset.get_temporal_node_features(timestep)

        return get_node_feats

    def get_sample(self, idx, test, **kwargs):
        """Get a training/test sample for the given time step.

        Args:
                idx: Time index for the sample.
                test: Whether this is a test sample (affects negative sampling).
                **kwargs: Additional arguments (e.g., all_edges flag).

        Returns:
                Dict with historical adjacencies, node features, and labels.
        """
        hist_adj_list = []
        hist_ndFeats_list = []
        hist_mask_list = []
        existing_nodes = []
        for i in range(idx - self.args.num_hist_steps, idx + 1):
            cur_adj = tu.get_sp_adj(
                edges=self.data.edges,
                time=i,
                weighted=True,
                time_window=self.args.adj_mat_time_window,
            )

            if self.args.smart_neg_sampling:
                existing_nodes.append(cur_adj["idx"].unique())
            else:
                existing_nodes = None

            node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

            # Get temporal features up to timestep i (features evolve over time)
            node_feats = self.get_node_feats(cur_adj, timestep=i)

            # Detach features to avoid keeping computation graph across timesteps
            node_feats = node_feats.detach()

            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # This would be if we were training on all the edges in the time_window
        label_adj = tu.get_sp_adj(
            edges=self.data.edges,
            time=idx + 1,
            weighted=False,
            time_window=self.args.adj_mat_time_window,
        )
        if test:
            neg_mult = self.args.negative_mult_test
        else:
            neg_mult = self.args.negative_mult_training

        if self.args.smart_neg_sampling:
            existing_nodes = torch.cat(existing_nodes)

        if "all_edges" in kwargs.keys() and kwargs["all_edges"] == True:
            non_exisiting_adj = tu.get_all_non_existing_edges(
                adj=label_adj, tot_nodes=self.data.num_nodes
            )
        else:
            non_exisiting_adj = tu.get_non_existing_edges(
                adj=label_adj,
                number=label_adj["vals"].size(0) * neg_mult,
                tot_nodes=self.data.num_nodes,
                smart_sampling=self.args.smart_neg_sampling,
                existing_nodes=existing_nodes,
            )

        # label_adj = tu.get_sp_adj_only_new(edges = self.data.edges,
        # 								   weighted = False,
        # 								   time = idx)

        label_adj["idx"] = torch.cat([label_adj["idx"], non_exisiting_adj["idx"]])
        label_adj["vals"] = torch.cat([label_adj["vals"], non_exisiting_adj["vals"]])
        return {
            "idx": idx,
            "hist_adj_list": hist_adj_list,
            "hist_ndFeats_list": hist_ndFeats_list,
            "label_sp": label_adj,
            "node_mask_list": hist_mask_list,
        }
