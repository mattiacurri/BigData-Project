"""Create and save per-user BERT-based embeddings for GAB datasets."""

import os
import pickle

import pandas as pd
import torch

import utils as u


def create_gab_embeddings(args):
    """Build and persist node feature lists (BERT) for GAB dataset in `args.gab_args.folder`."""
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

    # Filter out self-follows
    valid_edges = data[:, 0] != data[:, 1]
    data = data[valid_edges]

    # Remap node ids
    all_nodes = data[:, :2].flatten()
    unique_nodes = torch.unique(all_nodes)
    node_map = torch.zeros(unique_nodes.max() + 1, dtype=torch.long)
    node_map[unique_nodes] = torch.arange(len(unique_nodes))
    data[:, 0] = node_map[data[:, 0]]
    data[:, 1] = node_map[data[:, 1]]
    num_nodes = len(unique_nodes)
    unique_nodes_set = set(unique_nodes.tolist())

    # Load BERT embeddings
    emb_path = os.path.join(folder, "bert_features_real_posts.pkl")
    with open(emb_path, "rb") as f:
        post_emb = pickle.load(f)  # {post_id: emb}

    feats_per_node = 768
    node_feats_list = [torch.zeros(num_nodes, feats_per_node)]  # for time -1
    user_embs_accum = {}  # user_orig -> list of all embs up to current time
    for i, f in enumerate(folders):
        posts_path = os.path.join(folder, f, "posts_incremental.csv")
        if os.path.exists(posts_path):
            posts_df = pd.read_csv(posts_path)
            for _, row in posts_df.iterrows():
                post_id = row["id"]
                user_orig = row["account_id"]
                if post_id in post_emb:
                    if user_orig not in user_embs_accum:
                        user_embs_accum[user_orig] = []
                    user_embs_accum[user_orig].append(post_emb[post_id])
        feats_t = torch.zeros(num_nodes, feats_per_node)
        for user_orig, embs in user_embs_accum.items():
            if embs and user_orig in unique_nodes_set:
                avg_emb = torch.stack(embs).mean(dim=0)
                new_id = node_map[user_orig]
                feats_t[new_id] = avg_emb
        node_feats_list.append(feats_t)

    # Save the node_feats_list
    with open(os.path.join(folder, "node_feats_list.pkl"), "wb") as f:
        pickle.dump(node_feats_list, f)
    print(f"Saved node_feats_list to {os.path.join(folder, 'node_feats_list.pkl')}")


if __name__ == "__main__":
    args = u.Namespace({"gab_args": {"folder": "./data/gabdataset"}})
    create_gab_embeddings(args)
