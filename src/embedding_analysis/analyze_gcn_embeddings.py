"""Analyze GCN-produced embeddings for real vs synthetic nodes."""

import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from GabDataset import GabDataset
import modeling.egcn_h as egcn_h
import taskers_utils as tu
import utils as u


def load_model(model_path, device, config=None):
    """Load a GCN model checkpoint and return (gcn, args)."""
    args = SimpleNamespace(
        feats_per_node=768,
        layer_1_feats=128,
        layer_2_feats=64,
        k_top_grcu=768,
        num_layers=2,
        lstm_l1_layers=1,
        lstm_l1_feats=128,
        lstm_l2_layers=1,
        lstm_l2_feats=128,
        gcn_parameters={"cls_feats": 512},
    )

    gcn = egcn_h.EGCN(args, activation=nn.RReLU(), device=device)
    checkpoint = torch.load(model_path, map_location=device)
    gcn.load_state_dict(checkpoint["gcn_dict"])
    gcn = gcn.to(device)
    gcn.eval()
    return gcn, args


def prepare_graph_input(dataset, max_snapshot, device):
    """Prepare adj/feature/mask inputs for `max_snapshot` and move them to `device`."""
    adj = tu.get_sp_adj(edges=dataset.edges, time=max_snapshot, time_window=None)
    features = dataset.get_temporal_node_features(max_snapshot)
    num_nodes = dataset.num_nodes

    adj_norm = tu.normalize_adj(adj, num_nodes)
    mask = tu.get_node_mask(adj_norm, num_nodes)

    adj_batched = {
        "idx": adj_norm["idx"].unsqueeze(0),
        "vals": adj_norm["vals"].unsqueeze(0),
    }
    adj_tensor = u.sparse_prepare_tensor(adj_batched, [num_nodes]).to(device)

    hist_adj_list = [adj_tensor]
    hist_ndFeats_list = [features.to(device)]
    mask_list = [mask.to(device)]

    return hist_adj_list, hist_ndFeats_list, mask_list, num_nodes


def compute_gcn_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list):
    """Compute GCN node embeddings and return as a NumPy array."""
    with torch.inference_mode():
        nodes_embs = gcn(hist_adj_list, hist_ndFeats_list, mask_list)
    return nodes_embs.detach().cpu().numpy()


def main():
    """CLI entrypoint for GCN embeddings analysis and saving results/plots."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("GCN EMBEDDINGS ANALYSIS")
    print("=" * 60)

    print("\nLoading model...")
    model_path = project_root / "log/gab_h_c1_1_50/checkpoint_phase_4_best.pth.tar"
    gcn, args = load_model(str(model_path), device)
    print(f"GCN embedding dim: {args.layer_2_feats}")

    print("\nLoading real dataset...")
    real_args = SimpleNamespace(
        gab_args={"folder": str(project_root / "data/") + "/", "feats_per_node": 768}
    )
    dataset = GabDataset(real_args)
    max_snapshot = int(dataset.max_time.item())

    print("\nPreparing graph input...")
    hist_adj_list, hist_ndFeats_list, mask_list, num_real_nodes = prepare_graph_input(
        dataset, max_snapshot, device
    )

    print(f"\nComputing GCN embeddings for {num_real_nodes} real nodes...")
    real_gcn_embs = compute_gcn_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list)
    print(f"Real GCN embeddings shape: {real_gcn_embs.shape}")

    print("\nLoading synthetic data...")
    with open(project_root / "data/raw/bert_features_synthetic.pkl", "rb") as f:
        synth_bert_emb = pickle.load(f)

    batch_files = [
        project_root / "data/raw/batch1_synthetic.csv",
        project_root / "data/raw/batch2_synthetic.csv",
        project_root / "data/raw/batch3_synthetic.csv",
    ]

    import pandas as pd

    all_users = {}
    for bf in batch_files:
        df = pd.read_csv(bf)
        for user_id, group in df.groupby("user_id"):
            if user_id not in all_users:
                all_users[user_id] = []
            all_users[user_id].extend(group["post_id"].tolist())

    print(f"Total synthetic users: {len(all_users)}")

    synth_bert_embs = []
    for user_id in sorted(all_users.keys()):
        post_ids = all_users[user_id]
        embeddings = []
        for pid in post_ids:
            if pid in synth_bert_emb:
                emb = synth_bert_emb[pid]
                embeddings.append(emb.numpy() if hasattr(emb, "numpy") else emb)
        if embeddings:
            user_emb = np.array(embeddings).mean(axis=0)
            synth_bert_embs.append(user_emb)

    synth_bert_embs = np.array(synth_bert_embs)
    print(f"Synthetic BERT embeddings shape: {synth_bert_embs.shape}")

    print("\nBuilding hybrid graph with synthetic nodes...")

    synth_features = []
    for user_id in sorted(all_users.keys()):
        post_ids = all_users[user_id]
        embeddings = []
        for pid in post_ids:
            if pid in synth_bert_emb:
                emb = synth_bert_emb[pid]
                embeddings.append(emb.numpy() if hasattr(emb, "numpy") else emb)
        if embeddings:
            synth_features.append(np.array(embeddings).mean(axis=0))

    synth_features = np.array(synth_features)
    feat_matrix = np.vstack(
        [dataset.get_temporal_node_features(max_snapshot).numpy(), synth_features]
    )
    total_nodes = num_real_nodes + len(synth_features)

    real_adj = tu.get_sp_adj(edges=dataset.edges, time=max_snapshot, time_window=None)
    real_idx = real_adj["idx"]
    real_vals = real_adj["vals"]

    synth_ids = torch.arange(num_real_nodes, total_nodes)
    self_loops = torch.stack([synth_ids, synth_ids], dim=1)
    all_idx = torch.cat([real_idx, self_loops], dim=0)
    all_vals = torch.cat([real_vals, torch.ones(len(synth_ids), dtype=real_vals.dtype)])

    hybrid_adj = {"idx": all_idx, "vals": all_vals}
    adj_norm = tu.normalize_adj(hybrid_adj, total_nodes)
    mask = tu.get_node_mask(adj_norm, total_nodes)

    adj_batched = {
        "idx": adj_norm["idx"].unsqueeze(0),
        "vals": adj_norm["vals"].unsqueeze(0),
    }
    adj_tensor = u.sparse_prepare_tensor(adj_batched, [total_nodes]).to(device)

    hist_adj_list_synth = [adj_tensor]
    hist_ndFeats_list_synth = [torch.tensor(feat_matrix, dtype=torch.float).to(device)]
    mask_list_synth = [mask.to(device)]

    print(f"\nComputing GCN embeddings for hybrid graph ({total_nodes} nodes)...")
    all_gcn_embs = compute_gcn_embeddings(
        gcn, hist_adj_list_synth, hist_ndFeats_list_synth, mask_list_synth
    )

    real_gcn_embs_final = all_gcn_embs[:num_real_nodes]
    synth_gcn_embs = all_gcn_embs[num_real_nodes:]
    print(f"Real GCN embeddings shape: {real_gcn_embs_final.shape}")
    print(f"Synthetic GCN embeddings shape: {synth_gcn_embs.shape}")

    stats = {
        "gcn_embedding_dim": int(args.layer_2_feats),
        "num_real_nodes": int(num_real_nodes),
        "num_synthetic_nodes": len(synth_gcn_embs),
    }

    print("\n" + "=" * 60)
    print("REAL GCN EMBEDDINGS STATISTICS")
    print("=" * 60)

    n_samples = min(1000, len(real_gcn_embs_final))
    real_sample = real_gcn_embs_final[
        np.random.choice(len(real_gcn_embs_final), n_samples, replace=False)
    ]

    real_cos = 1 - pdist(real_sample, metric="cosine")
    real_euc = pdist(real_sample, metric="euclidean")

    print(f"\nCosine Similarity (Real GCN):")
    print(f"  Mean: {real_cos.mean():.4f}")
    print(f"  Std: {real_cos.std():.4f}")
    print(f"  Min: {real_cos.min():.4f}")
    print(f"  Max: {real_cos.max():.4f}")

    stats["real_gcn"] = {
        "cosine_similarity": {
            "mean": float(real_cos.mean()),
            "std": float(real_cos.std()),
            "min": float(real_cos.min()),
            "max": float(real_cos.max()),
        },
        "embedding_variance": float(real_gcn_embs_final.var()),
    }

    print("\n" + "=" * 60)
    print("SYNTHETIC GCN EMBEDDINGS STATISTICS")
    print("=" * 60)

    n_samples_synth = min(1000, len(synth_gcn_embs))
    synth_sample = synth_gcn_embs[
        np.random.choice(len(synth_gcn_embs), n_samples_synth, replace=False)
    ]

    synth_cos = 1 - pdist(synth_sample, metric="cosine")
    synth_euc = pdist(synth_sample, metric="euclidean")

    print(f"\nCosine Similarity (Synthetic GCN):")
    print(f"  Mean: {synth_cos.mean():.4f}")
    print(f"  Std: {synth_cos.std():.4f}")
    print(f"  Min: {synth_cos.min():.4f}")
    print(f"  Max: {synth_cos.max():.4f}")

    stats["synthetic_gcn"] = {
        "cosine_similarity": {
            "mean": float(synth_cos.mean()),
            "std": float(synth_cos.std()),
            "min": float(synth_cos.min()),
            "max": float(synth_cos.max()),
        },
        "embedding_variance": float(synth_gcn_embs.var()),
    }

    print("\n" + "=" * 60)
    print("CROSS-GROUP COMPARISON")
    print("=" * 60)

    cross_cos = 1 - cdist(real_sample, synth_sample, metric="cosine")
    print(f"\nReal-Synth Cosine Similarity:")
    print(f"  Mean: {cross_cos.mean():.4f}")
    print(f"  Std: {cross_cos.std():.4f}")

    print(f"\nVariance Comparison:")
    real_var = real_gcn_embs_final.var(axis=0).mean()
    synth_var = synth_gcn_embs.var(axis=0).mean()
    print(f"  Real variance: {real_var:.6f}")
    print(f"  Synthetic variance: {synth_var:.6f}")
    print(f"  Ratio (real/synth): {real_var / synth_var:.2f}x")

    stats["cross_group"] = {
        "real_synth_cosine_similarity": {
            "mean": float(cross_cos.mean()),
            "std": float(cross_cos.std()),
        },
        "variance_ratio": float(real_var / synth_var),
    }

    stats["comparison_summary"] = {
        "real_cosine_mean": float(real_cos.mean()),
        "synth_cosine_mean": float(synth_cos.mean()),
        "difference": float(synth_cos.mean() - real_cos.mean()),
        "synth_more_similar": bool(synth_cos.mean() > real_cos.mean()),
    }

    with open(output_dir / "gcn_embeddings_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {output_dir / 'gcn_embeddings_stats.json'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(real_cos, bins=50, alpha=0.7, label="Real GCN", density=True)
    ax.hist(synth_cos, bins=50, alpha=0.7, label="Synthetic GCN", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("GCN Embeddings: Cosine Similarity Distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(cross_cos.flatten(), bins=50, alpha=0.7, color="green", density=True)
    ax.axvline(
        cross_cos.mean(), color="red", linestyle="--", label=f"Mean: {cross_cos.mean():.3f}"
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cross-group: Real vs Synthetic GCN Similarity")
    ax.legend()

    ax = axes[1, 0]
    ax.bar(
        ["Real", "Synthetic"],
        [real_cos.mean(), synth_cos.mean()],
        yerr=[real_cos.std(), synth_cos.std()],
        capsize=5,
    )
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Similarity: Real vs Synthetic GCN")

    ax = axes[1, 1]
    real_vars_per_dim = real_gcn_embs_final.var(axis=0)
    synth_vars_per_dim = synth_gcn_embs.var(axis=0)
    ax.plot(real_vars_per_dim, alpha=0.7, label="Real")
    ax.plot(synth_vars_per_dim, alpha=0.7, label="Synthetic")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Variance")
    ax.set_title("Variance per Dimension: GCN Embeddings")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "gcn_embeddings_comparison.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'gcn_embeddings_comparison.png'}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(real_euc, bins=50, alpha=0.7, label="Real", density=True)
    plt.hist(synth_euc, bins=50, alpha=0.7, label="Synthetic", density=True)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("GCN Embeddings: Euclidean Distance Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(
        ["Real-Real", "Synth-Synth", "Real-Synth"],
        [
            real_euc.mean(),
            synth_euc.mean(),
            cdist(real_sample, synth_sample, metric="euclidean").mean(),
        ],
    )
    plt.ylabel("Mean Euclidean Distance")
    plt.title("Distance Comparison")

    plt.tight_layout()
    plt.savefig(output_dir / "gcn_embeddings_distances.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'gcn_embeddings_distances.png'}")

    np.save(output_dir / "real_gcn_embeddings.npy", real_gcn_embs_final)
    np.save(output_dir / "synth_gcn_embeddings.npy", synth_gcn_embs)
    print(f"Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()
