"""Link prediction tasker for temporal graph learning.

Handles the preparation and sampling of data for link prediction tasks on temporal graphs.
"""

import torch

import taskers_utils as tu


class LinkPrediction:
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
        self.dataset = dataset
        self.args = args
        self.nodes = dataset.feats_per_node

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

    def _get_node_features_for_active_nodes(
        self, timestep, active_node_indices, max_allowed_timestep=None
    ):
        """Get node features only for active nodes.

        Args:
            timestep: Current timestep. Negative timesteps are clamped to 0.
            active_node_indices: Tensor of active node indices.
            max_allowed_timestep: Maximum timestep to use for features (to avoid data leakage in test).

        Returns:
            Node features tensor of shape [num_active_nodes, feat_dim].
        """
        # Clamp negative timesteps to 0 - negative timesteps have no data anyway
        effective_timestep = max(0, timestep)
        if max_allowed_timestep is not None:
            effective_timestep = min(effective_timestep, max_allowed_timestep)

        # Get full features
        full_feats = self.dataset.get_temporal_node_features(effective_timestep)

        # Select only active nodes
        return full_feats[active_node_indices]

    def get_sample(self, idx, test, **kwargs):
        """Get a training/test sample for the given time step.

        Args:
                idx: Time index for the sample.
                test: Whether this is a test sample (affects negative sampling).
                **kwargs: Additional arguments (e.g., all_edges flag).

        Returns:
                Dict with historical adjacencies, node features, and labels.
        """
        hist_adj_list = []  # List of historical adjacency matrices
        hist_ndFeats_list = []  # List of historical node features
        hist_mask_list = []  # List of node masks
        existing_nodes = []  # List of existing nodes for smart negative sampling

        # Get the number of nodes that exist at this snapshot (for memory efficiency)
        # Use only active nodes to reduce memory footprint significantly
        num_active_nodes = len(self.dataset.cumulative_nodes_per_snapshot[idx])
        active_node_indices = self.dataset.get_node_indices_at_snapshot(idx)

        # Create a mapping from original node IDs to compacted IDs
        # This allows us to work with smaller tensors
        node_mapping = {
            int(orig_idx): compact_idx
            for compact_idx, orig_idx in enumerate(active_node_indices.tolist())
        }

        # Clamp start index to 0 - negative timesteps don't have data
        start_time = max(0, idx - self.args.num_hist_steps)

        # Collect historical snapshots.
        # For every snapshot we take the adjacency matrix, the node features, and the node mask to distinguish nodes that are in that snapshot from those that are not (since we are using a compacted representation with only active nodes, the mask is needed to distinguish between "inactive" nodes that are not present in that snapshot vs "active" nodes that are present but just have no edges).
        for i in range(start_time, idx):
            # get the adjacency matrix
            cur_adj = tu.get_sp_adj(
                edges=self.dataset.edges,
                time=i,
                time_window=self.args.adj_mat_time_window,
            )

            # Remap existing nodes to compacted space
            unique_nodes = cur_adj["idx"].unique()
            remapped = [
                node_mapping[int(n)] for n in unique_nodes.tolist() if int(n) in node_mapping
            ]
            if remapped:
                existing_nodes.append(torch.tensor(remapped, dtype=torch.long))

            # Remap adjacency indices to compacted space
            cur_adj = self._remap_adj_to_active_nodes(cur_adj, node_mapping, num_active_nodes)

            # Use compacted number of nodes
            node_mask = tu.get_node_mask(cur_adj, num_active_nodes)

            # Get temporal features for active nodes only - use all info up to current snapshot
            node_feats = self._get_node_features_for_active_nodes(
                i, active_node_indices, max_allowed_timestep=None
            )

            # Detach features to avoid keeping computation graph across timesteps
            node_feats = node_feats.detach()

            # Use compacted number of nodes
            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=num_active_nodes)

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # AUTOMATIC COLD START: If no history available (first snapshot), use empty graph
        # This happens when idx=0 or when idx - num_hist_steps < 0
        if len(hist_adj_list) == 0:
            # Create empty adjacency (only self-loops after normalization)
            empty_adj = {
                "idx": torch.zeros((0, 2), dtype=torch.long),
                "vals": torch.zeros(0, dtype=torch.long),
            }

            empty_adj = tu.normalize_adj(adj=empty_adj, num_nodes=num_active_nodes)

            # Use features from previous snapshot (or ZEROS if idx=0)
            if idx == 0:
                # Truly cold start - no features available, use zero vectors
                node_feats = torch.zeros(
                    num_active_nodes, self.dataset.feats_per_node, dtype=torch.float32
                )
            else:
                node_feats = self._get_node_features_for_active_nodes(
                    idx - 1, active_node_indices, max_allowed_timestep=None
                )
                node_feats = node_feats.detach()

            node_mask = tu.get_node_mask(empty_adj, num_active_nodes)

            hist_adj_list.append(empty_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # Get label edges for the current timestep
        # NOTE: Since the dataset is incremental (cumulative), all edges at time idx
        # include both old and new edges. To test only on NEW edges, we need to filter.
        label_adj_current = tu.get_sp_adj(
            edges=self.dataset.edges,
            time=idx,
            time_window=1,
        )

        # Get edges from previous snapshot to compute the difference
        if idx > 0:
            label_adj_previous = tu.get_sp_adj(
                edges=self.dataset.edges,
                time=idx - 1,
                time_window=1,
            )

            # Create sets of edges for efficient comparison
            current_edges = set(tuple(edge.tolist()) for edge in label_adj_current["idx"])
            previous_edges = set(tuple(edge.tolist()) for edge in label_adj_previous["idx"])

            # Compute NEW edges (present in current but not in previous)
            new_edges_set = current_edges - previous_edges

            if len(new_edges_set) > 0:
                # Convert back to tensor
                new_edges_list = list(new_edges_set)
                label_adj_new = {
                    "idx": torch.tensor(new_edges_list, dtype=torch.long),
                    "vals": torch.ones(len(new_edges_list), dtype=torch.long),
                }
            else:
                # No new edges - return empty
                label_adj_new = {
                    "idx": torch.zeros((0, 2), dtype=torch.long),
                    "vals": torch.zeros(0, dtype=torch.long),
                }

            # For negative sampling: combine previous AND current edges
            # This ensures we don't sample edges that existed in training (t-1) OR in current (t)
            all_edges_union = current_edges | previous_edges
            all_edges_list = list(all_edges_union)
            label_adj_all_combined = {
                "idx": torch.tensor(all_edges_list, dtype=torch.long),
                "vals": torch.ones(len(all_edges_list), dtype=torch.long),
            }
        else:
            # First snapshot - all edges are new
            label_adj_new = label_adj_current
            label_adj_all_combined = label_adj_current

        # Remap to compacted space
        # Keep TWO versions:
        # 1. label_adj_new: only NEW edges (for positive samples)
        # 2. label_adj_all: ALL edges from both t-1 and t (for negative sampling - to avoid)
        label_adj_new = self._remap_adj_to_active_nodes(
            label_adj_new, node_mapping, num_active_nodes
        )

        # For negative sampling, we need to avoid ALL edges (from t-1 AND t)
        # This ensures negative samples are "pure" - never connected before
        label_adj_all = self._remap_adj_to_active_nodes(
            label_adj_all_combined, node_mapping, num_active_nodes
        )
        if test:
            negative_multiplier = self.args.negative_mult_test
        else:
            negative_multiplier = self.args.negative_mult_training

        # Determine which nodes to use for negative sampling
        if test:
            # TEST: Use ALL active nodes (including new nodes appearing at time t)
            # This ensures we evaluate the model's ability to predict edges to/from new nodes
            existing_nodes = torch.arange(num_active_nodes, dtype=torch.long)
        else:
            # TRAIN: Use only historical nodes (appeared in previous snapshots)
            # This creates more realistic negative samples based on historical activity
            if len(existing_nodes) == 0:
                # Cold start: no history, use all active nodes
                existing_nodes = torch.arange(num_active_nodes, dtype=torch.long)
            else:
                existing_nodes = torch.cat(existing_nodes)

        # Check if label_adj_new has any edges after remapping
        # If not, we can't generate meaningful negative samples from the positive distribution
        # if label_adj_new["idx"].size(0) == 0:
        #     print(f"WARNING: No label edges for snapshot {idx} after remapping!")
        #     print(
        #         f"  num_active_nodes: {num_active_nodes}, cumulative nodes: {len(self.dataset.cumulative_nodes_per_snapshot.get(idx, set()))}"
        #     )
        #     # Return empty sample - cannot train without any positive/negative examples
        #     return {
        #         "idx": idx,
        #         "hist_adj_list": hist_adj_list,
        #         "hist_ndFeats_list": hist_ndFeats_list,
        #         "label_sp": {
        #             "idx": torch.zeros((0, 2), dtype=torch.long),
        #             "vals": torch.zeros(0, dtype=torch.long),
        #         },
        #         "node_mask_list": hist_mask_list,
        #     }

        # Generate negative samples
        # IMPORTANT: Use label_adj_all (all edges, old+new) to avoid sampling
        # edges that were already in the training set!
        if "all_edges" in kwargs and kwargs["all_edges"]:
            non_existing_adj = tu.get_all_non_existing_edges(
                adj=label_adj_all, tot_nodes=num_active_nodes
            )
        else:
            non_existing_adj = tu.get_non_existing_edges(
                adj=label_adj_all,  # Use ALL edges to avoid old+new
                number=label_adj_new["vals"].size(0)
                * negative_multiplier,  # Number based on NEW edges
                tot_nodes=num_active_nodes,
                existing_nodes=existing_nodes,
            )

        # Combine NEW positive edges with negative edges
        label_adj_new["idx"] = torch.cat([label_adj_new["idx"], non_existing_adj["idx"]])
        label_adj_new["vals"] = torch.cat([label_adj_new["vals"], non_existing_adj["vals"]])

        # For filtered ranking: include edges from previous snapshot (remapped)
        # These should be excluded from ranking to avoid penalizing correct predictions
        if idx > 0:
            label_adj_previous_remapped = self._remap_adj_to_active_nodes(
                label_adj_previous, node_mapping, num_active_nodes
            )
            filter_edges = label_adj_previous_remapped["idx"]
        else:
            # First snapshot: no edges to filter
            filter_edges = torch.zeros((0, 2), dtype=torch.long)

        # Statistics for logging
        # num_positive = (label_adj_new["vals"] == 1).sum().item()
        # num_negative = (label_adj_new["vals"] == 0).sum().item()
        # num_filter = filter_edges.size(0)

        # # Print statistics (only if positive edges exist)
        # if num_positive > 0:
        #     mode_str = "TEST" if test else "TRAIN"
        #     print(f"\n{'=' * 70}")
        #     print(f"{mode_str} Sample Statistics for Snapshot {idx}:")
        #     print(f"  Active nodes: {num_active_nodes:,}")
        #     print(f"  Positive edges (NEW): {num_positive:,}")
        #     print(f"  Negative edges: {num_negative:,}")
        #     print(f"  Filter edges (from previous snapshot): {num_filter:,}")
        #     if num_filter > 0:
        #         print(
        #             f"  → These {num_filter:,} edges from training will be excluded from MRR ranking"
        #         )
        #     print(f"{'=' * 70}\n")

        return {
            "idx": idx,
            "hist_adj_list": hist_adj_list,
            "hist_ndFeats_list": hist_ndFeats_list,
            "label_sp": label_adj_new,
            "node_mask_list": hist_mask_list,
            "filter_edges": filter_edges,  # Edges to exclude from ranking metrics
        }
