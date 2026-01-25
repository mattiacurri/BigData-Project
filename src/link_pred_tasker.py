"""Link prediction tasker for temporal graph learning.

Handles the preparation and sampling of data for link prediction tasks on temporal graphs.
"""

import torch

import taskers_utils as tu


class Link_Pred_Tasker:
    """Creates a tasker object which computes the required inputs for training on a link prediction task.

    It receives a dataset object which should have two attributes: nodes_feats and edges, this
    makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
    structure).

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

        # Use dataset features (BERT embeddings)
        self.feats_per_node = dataset.feats_per_node
        print(f"Using {self.feats_per_node} features per node")

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

    def _remap_adj_to_active_nodes(self, adj, node_mapping, num_active_nodes):
        """Remap adjacency matrix indices to compacted active node space.

        Args:
            adj: Adjacency dict with 'idx' and 'vals'.
            node_mapping: Dict mapping original node IDs to compacted IDs.
            num_active_nodes: Total number of active nodes.

        Returns:
            Remapped adjacency dict with compacted indices.
        """
        idx = adj["idx"]
        vals = adj["vals"]

        if idx.size(0) == 0:
            # No edges, return empty
            return {"idx": idx, "vals": vals}

        # Filter edges where both nodes are in the active set
        src_nodes = idx[:, 0].tolist()
        dst_nodes = idx[:, 1].tolist()

        new_src = []
        new_dst = []
        new_vals = []

        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            if src in node_mapping and dst in node_mapping:
                new_src.append(node_mapping[src])
                new_dst.append(node_mapping[dst])
                new_vals.append(vals[i].item())

        if len(new_src) == 0:
            # No valid edges after remapping
            return {
                "idx": torch.zeros((0, 2), dtype=torch.long),
                "vals": torch.zeros(0, dtype=vals.dtype),
            }

        new_idx = torch.tensor(list(zip(new_src, new_dst)), dtype=torch.long)
        new_vals = torch.tensor(new_vals, dtype=vals.dtype)

        return {"idx": new_idx, "vals": new_vals}

    def _get_node_features_for_active_nodes(self, timestep, active_node_indices):
        """Get node features only for active nodes.

        Args:
            timestep: Current timestep. Negative timesteps are clamped to 0.
            active_node_indices: Tensor of active node indices.

        Returns:
            Node features tensor of shape [num_active_nodes, feat_dim].
        """
        # Clamp negative timesteps to 0 - negative timesteps have no data anyway
        effective_timestep = max(0, timestep)

        # Get full features
        full_feats = self.data.get_temporal_node_features(effective_timestep)

        # Select only active nodes
        active_feats = full_feats[active_node_indices]

        return active_feats

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

        # Get the number of nodes that exist at this snapshot (for memory efficiency)
        # Use only active nodes to reduce memory footprint significantly
        num_active_nodes = self.data.get_num_nodes_at_snapshot(idx)
        print(f"Snapshot {idx}: {num_active_nodes} active nodes")
        active_node_indices = self.data.get_node_indices_at_snapshot(idx)
        print(f"Total nodes in dataset: {self.data.num_nodes}")

        # Create a mapping from original node IDs to compacted IDs
        # This allows us to work with smaller tensors
        node_mapping = {
            int(orig_idx): compact_idx
            for compact_idx, orig_idx in enumerate(active_node_indices.tolist())
        }

        # Clamp start index to 0 - negative timesteps don't have data
        start_time = max(0, idx - self.args.num_hist_steps)

        for i in range(start_time, idx + 1):
            cur_adj = tu.get_sp_adj(
                edges=self.data.edges,
                time=i,
                weighted=True,
                time_window=self.args.adj_mat_time_window,
            )

            if self.args.smart_neg_sampling:
                # Remap existing nodes to compacted space
                unique_nodes = cur_adj["idx"].unique()
                remapped = [
                    node_mapping[int(n)] for n in unique_nodes.tolist() if int(n) in node_mapping
                ]
                if remapped:
                    existing_nodes.append(torch.tensor(remapped, dtype=torch.long))
            else:
                existing_nodes = None

            # Remap adjacency indices to compacted space
            cur_adj = self._remap_adj_to_active_nodes(cur_adj, node_mapping, num_active_nodes)

            # Use compacted number of nodes
            node_mask = tu.get_node_mask(cur_adj, num_active_nodes)

            # Get temporal features for active nodes only
            node_feats = self._get_node_features_for_active_nodes(i, active_node_indices)

            # Detach features to avoid keeping computation graph across timesteps
            node_feats = node_feats.detach()

            # Use compacted number of nodes
            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=num_active_nodes)

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # Get label edges for the next timestep
        label_adj = tu.get_sp_adj(
            edges=self.data.edges,
            time=idx + 1,
            weighted=False,
            time_window=self.args.adj_mat_time_window,
        )

        # Remap label_adj to compacted space
        label_adj = self._remap_adj_to_active_nodes(label_adj, node_mapping, num_active_nodes)

        if test:
            neg_mult = self.args.negative_mult_test
        else:
            neg_mult = self.args.negative_mult_training

        if self.args.smart_neg_sampling and existing_nodes:
            existing_nodes = torch.cat(existing_nodes)
        else:
            existing_nodes = None

        # if "all_edges" in kwargs.keys() and kwargs["all_edges"] == True:
        #     # Use max_negative_test_edges if configured to limit memory usage
        #     max_neg = getattr(self.args, "max_negative_test_edges", None)
        #     non_existing_adj = tu.get_all_non_existing_edges(
        #         adj=label_adj, tot_nodes=num_active_nodes, max_edges=max_neg
        #     )
        # else:
        #     non_existing_adj = tu.get_non_existing_edges(
        #         adj=label_adj,
        #         number=label_adj["vals"].size(0) * neg_mult,
        #         tot_nodes=num_active_nodes,
        #         smart_sampling=self.args.smart_neg_sampling,
        #         existing_nodes=existing_nodes,
        #     )

        if "all_edges" in kwargs.keys() and kwargs["all_edges"] == True:
            non_existing_adj = tu.get_all_non_existing_edges(
                adj=label_adj, tot_nodes=num_active_nodes
            )
        else:
            non_existing_adj = tu.get_non_existing_edges(
                adj=label_adj,
                number=label_adj["vals"].size(0) * neg_mult,
                tot_nodes=num_active_nodes,
                smart_sampling=self.args.smart_neg_sampling,
                existing_nodes=existing_nodes,
            )

        label_adj["idx"] = torch.cat([label_adj["idx"], non_existing_adj["idx"]])
        label_adj["vals"] = torch.cat([label_adj["vals"], non_existing_adj["vals"]])

        return {
            "idx": idx,
            "hist_adj_list": hist_adj_list,
            "hist_ndFeats_list": hist_ndFeats_list,
            "label_sp": label_adj,
            "node_mask_list": hist_mask_list,
        }
