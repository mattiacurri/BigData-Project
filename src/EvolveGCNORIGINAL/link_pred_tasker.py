"""Tasker implementations for training/evaluating link-prediction models."""

import torch

import taskers_utils as tu
import utils as u


class Link_Pred_Tasker:
    """Tasker that builds inputs for link-prediction training and evaluation.

    Produces historical adjacency lists, node-features and label adjacency for a time step.
    """

    def __init__(self, args, dataset):
        """Initialize Link_Pred_Tasker with `args` and `dataset`."""
        self.data = dataset
        # max_time for link pred should be one before
        self.max_time = dataset.max_time - 1
        self.args = args
        self.num_classes = 2

        if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
            self.feats_per_node = dataset.feats_per_node

        self.get_node_feats = self.build_get_node_feats(args, dataset)
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)

        # Cache for preprocessed adjacencies and node features
        self.adj_cache = {}
        self.mask_cache = {}  # Cache for node masks (computed before normalization)
        self.feats_cache = {}
        # Cache for historical edges as tensor hashes (cumulative up to each time step)
        self.hist_edges_cache = {}
        self.num_nodes_hash_const = self.data.num_nodes  # For hash computation

        # Track used edges during test as tensor hashes
        self.used_edges = set()
        self.is_test_mode = False

    def build_prepare_node_feats(self, args, dataset):
        """Return a `prepare_node_feats` callable according to requested feature type."""
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:

            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(
                    node_feats, torch_size=[dataset.num_nodes, self.feats_per_node]
                )
        else:
            prepare_node_feats = self.data.prepare_node_feats

        return prepare_node_feats

    def build_get_node_feats(self, args, dataset):
        """Return a `get_node_feats` callable according to configured node-feature mode."""
        if args.use_2_hot_node_feats:
            max_deg_out, max_deg_in = tu.get_max_degs(args, dataset)
            self.feats_per_node = max_deg_out + max_deg_in

            def get_node_feats(adj, time):
                return tu.get_2_hot_deg_feats(adj, max_deg_out, max_deg_in, dataset.num_nodes)
        elif args.use_1_hot_node_feats:
            print("Using 1-hot degree-based node features")
            max_deg, _ = tu.get_max_degs(args, dataset)
            self.feats_per_node = max_deg

            def get_node_feats(adj, time):
                return tu.get_1_hot_deg_feats(adj, max_deg, dataset.num_nodes)
        else:

            def get_node_feats(adj, time):
                return dataset.nodes_feats

        return get_node_feats

    def get_sample(self, idx, test, **kwargs):
        """Build and return the training/evaluation sample for `idx` (time)."""
        hist_adj_list = []
        hist_ndFeats_list = []
        hist_mask_list = []
        existing_nodes = []

        # Reset used_edges at the start of test phase
        if test and not self.is_test_mode:
            self.is_test_mode = True
            self.used_edges = set()

        for i in range(idx - self.args.num_hist_steps, idx):
            # Check cache for adjacency and mask
            if i not in self.adj_cache:
                cur_adj = tu.get_sp_adj(
                    edges=self.data.edges,
                    time=i,
                    weighted=False,
                    time_window=self.args.adj_mat_time_window,
                    cumulative=True,
                )
                # IMPORTANT: Compute mask BEFORE normalization!
                # Normalization adds identity matrix (self-loops for ALL nodes),
                # which would make all nodes appear "active" in the mask.
                # We want the mask to reflect only nodes that have actual edges.
                node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)
                cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)

                self.adj_cache[i] = cur_adj
                self.mask_cache[i] = node_mask
            else:
                cur_adj = self.adj_cache[i]
                node_mask = self.mask_cache[i]

            if self.args.smart_neg_sampling:
                existing_nodes.append(cur_adj["idx"].unique())
            else:
                existing_nodes = None

            # Check cache for node features
            if i not in self.feats_cache:
                if self.args.use_2_hot_node_feats or self.args.use_1_hot_node_feats:
                    node_feats = self.get_node_feats(cur_adj, i)
                else:
                    node_feats = self.data.get_node_feats(i)
                self.feats_cache[i] = node_feats
            else:
                node_feats = self.feats_cache[i]

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)

        # This would be if we were training on all the edges in the time_window
        label_adj = tu.get_sp_adj(
            edges=self.data.edges,
            time=idx,
            weighted=False,
            time_window=1,
            cumulative=False,
        )

        # Filter out edges that are already in historical data (all past edges)
        # Use cache for hist_edges as tensor hashes - build cumulatively
        if idx not in self.hist_edges_cache:
            # Build hist_edges cumulatively from previous cached value
            if idx > self.data.min_time and (idx - 1) in self.hist_edges_cache:
                prev_hashes = self.hist_edges_cache[idx - 1]
                # Add edges from time idx-1 using adj_cache if available
                if idx - 1 in self.adj_cache:
                    past_adj = self.adj_cache[idx - 1]
                else:
                    past_adj = tu.get_sp_adj(
                        edges=self.data.edges,
                        time=idx - 1,
                        weighted=False,
                        time_window=1,
                        cumulative=True,
                    )
                # Convert edges to hashes on CPU (for set operations, then back to tensor)
                new_edges = past_adj["idx"]
                new_hashes = new_edges[:, 0] * self.num_nodes_hash_const + new_edges[:, 1]
                hist_hashes = torch.unique(torch.cat([prev_hashes, new_hashes]))
            else:
                # Build from scratch (first time or no previous cache)
                all_hashes = []
                for t in range(self.data.min_time, idx):
                    if t in self.adj_cache:
                        past_adj = self.adj_cache[t]
                    else:
                        past_adj = tu.get_sp_adj(
                            edges=self.data.edges,
                            time=t,
                            weighted=False,
                            time_window=1,
                            cumulative=True,
                        )
                    edges = past_adj["idx"]
                    edge_hashes = edges[:, 0] * self.num_nodes_hash_const + edges[:, 1]
                    all_hashes.append(edge_hashes)
                if all_hashes:
                    hist_hashes = torch.unique(torch.cat(all_hashes))
                else:
                    hist_hashes = torch.tensor([], dtype=torch.long)
            self.hist_edges_cache[idx] = hist_hashes
        else:
            hist_hashes = self.hist_edges_cache[idx]

        # Filter label edges using vectorized torch.isin
        label_edges = label_adj["idx"]
        label_hashes = label_edges[:, 0] * self.num_nodes_hash_const + label_edges[:, 1]
        mask = ~torch.isin(label_hashes, hist_hashes)
        label_adj["idx"] = label_adj["idx"][mask]
        label_adj["vals"] = label_adj["vals"][mask]

        if test:
            neg_mult = self.args.negative_mult_test
            # Add positive edges to used_edges for test tracking (keep as set for union)
            new_label_hashes = (
                label_adj["idx"][:, 0] * self.num_nodes_hash_const + label_adj["idx"][:, 1]
            )
            for h in new_label_hashes.tolist():
                self.used_edges.add(h)
        else:
            neg_mult = self.args.negative_mult_training

        if self.args.smart_neg_sampling:
            existing_nodes = torch.cat(existing_nodes) if existing_nodes else None

        # For test: forbid edges already used in previous test samples
        # Combine hist_hashes with used_edges
        if test and self.used_edges:
            used_tensor = torch.tensor(list(self.used_edges), dtype=hist_hashes.dtype)
            all_forbidden_hashes = torch.unique(torch.cat([hist_hashes, used_tensor]))
        else:
            all_forbidden_hashes = hist_hashes

        non_exisiting_adj = tu.get_non_existing_edges(
            adj=label_adj,
            number=label_adj["vals"].size(0) * neg_mult,
            tot_nodes=self.data.num_nodes,
            smart_sampling=self.args.smart_neg_sampling,
            existing_nodes=existing_nodes,
            forbidden_edge_hashes=all_forbidden_hashes,
            num_nodes_hash_const=self.num_nodes_hash_const,
        )

        # Add sampled negative edges to used_edges for test tracking
        if test:
            neg_hashes = (
                non_exisiting_adj["idx"][:, 0] * self.num_nodes_hash_const
                + non_exisiting_adj["idx"][:, 1]
            )
            for h in neg_hashes.tolist():
                self.used_edges.add(h)

        label_adj["idx"] = torch.cat([label_adj["idx"], non_exisiting_adj["idx"]])
        label_adj["vals"] = torch.cat([label_adj["vals"], non_exisiting_adj["vals"]])
        # Return hist_hashes for filter_edges (convert to list of tuples for backward compatibility)
        filter_edges = [
            (h // self.num_nodes_hash_const, h % self.num_nodes_hash_const)
            for h in hist_hashes.tolist()
        ]
        return {
            "idx": idx,
            "hist_adj_list": hist_adj_list,
            "hist_ndFeats_list": hist_ndFeats_list,
            "label_sp": label_adj,
            "node_mask_list": hist_mask_list,
            "filter_edges": filter_edges,
        }
