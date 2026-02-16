"""Dataset loader for GAB collections and node-feature preparation."""

import os
import pickle

import pandas as pd
import torch

import utils as u


class Gab:
    """Helper to load GAB datasets and optional BERT-based node features."""

    def __init__(self, args):
        """Load dataset files from `args.gab_args.folder` and prepare node features."""
        args.gab_args = u.Namespace(args.gab_args)

        folder = args.gab_args.folder
        folders = [
            f
            for f in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, f)) and "synthetic" not in f
        ]
        all_data = []
        for i, f in enumerate(folders):
            path = os.path.join(folder, f, "social_network.edg")
            df = pd.read_csv(path, sep="\t", header=None, names=["source", "target"])
            data = torch.tensor(df.values, dtype=torch.long)
            weight_col = torch.ones((data.shape[0], 1), dtype=torch.long)
            time_col = torch.full((data.shape[0], 1), i, dtype=torch.long)
            data = torch.cat([data, weight_col, time_col], dim=1)
            all_data.append(data)
        data = torch.cat(all_data, dim=0)

        # Filter out self-follows (edges where source == target)
        valid_edges = data[:, 0] != data[:, 1]
        data = data[valid_edges]

        # Remap node ids to 0 to num_nodes-1
        all_nodes = data[:, :2].flatten()
        unique_nodes = torch.unique(all_nodes)
        node_map = torch.zeros(unique_nodes.max() + 1, dtype=torch.long)
        node_map[unique_nodes] = torch.arange(len(unique_nodes))
        data[:, 0] = node_map[data[:, 0]]
        data[:, 1] = node_map[data[:, 1]]
        self.num_nodes = len(unique_nodes)
        self.contID_to_origID = unique_nodes.tolist()
        unique_nodes_set = set(unique_nodes.tolist())

        self.edges = self.load_edges(args, data)

        # Load BERT embeddings only if using external features
        if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
            print("Using BERT-based node features")
            self.feats_per_node = 768
            self.prepare_node_feats = lambda x: x  # dense features
            emb_path = os.path.join(folder, "bert_features_real_posts.pkl")
            with open(emb_path, "rb") as f:
                self.post_emb = pickle.load(f)  # {post_id: emb}

            # Compute node features per time
            self.node_feats_list = [
                torch.zeros(self.num_nodes, self.feats_per_node)
            ]  # for time -1
            user_embs_accum = {}  # user_orig -> list of all embs up to current time
            for i, f in enumerate(folders):
                posts_path = os.path.join(folder, f, "posts_incremental.csv")
                if os.path.exists(posts_path):
                    posts_df = pd.read_csv(posts_path)
                    for _, row in posts_df.iterrows():
                        post_id = row["id"]
                        user_orig = row["account_id"]
                        if post_id in self.post_emb:
                            if user_orig not in user_embs_accum:
                                user_embs_accum[user_orig] = []
                            user_embs_accum[user_orig].append(self.post_emb[post_id])
                    feats_t = torch.zeros(self.num_nodes, self.feats_per_node)
                    for user_orig, embs in user_embs_accum.items():
                        if embs and user_orig in unique_nodes_set:
                            avg_emb = torch.stack(embs).mean(dim=0)
                            new_id = node_map[user_orig]
                            feats_t[new_id] = avg_emb
                    self.node_feats_list.append(feats_t)
        else:
            print("Using degree-based node features")
            # For degree-based features, node_feats_list is not used
            self.node_feats_list = None

    def get_node_feats(self, time):
        """Return node features for the requested `time` snapshot."""
        if self.node_feats_list is None:
            raise ValueError(
                "Node features not loaded. Use embedding features or degree-based features."
            )
        if time < self.min_time - 1:
            return self.node_feats_list[0]
        if time > self.max_time:
            return self.node_feats_list[-1]
        idx = time - int(self.min_time) + 1
        return self.node_feats_list[idx]

    def load_edges(self, args, data):
        """Load `data` (source, target, weight, time) into internal edge dict format."""
        # data is already loaded as tensor with columns: source, target, weight, time
        cols = u.Namespace({"source": 0, "target": 1, "weight": 2, "time": 3})

        data = data.long()

        ids = data[:, cols.source] * self.num_nodes + data[:, cols.target]
        self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))

        idx = data[:, [cols.source, cols.target, cols.time]]

        self.max_time = data[:, cols.time].max()
        self.min_time = data[:, cols.time].min()

        return {"idx": idx, "vals": torch.ones(idx.size(0))}
