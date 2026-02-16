"""Analyze GCN embeddings from multiple models."""

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from GabDataset import GabDataset
import modeling.egcn_h as egcn_h
import taskers_utils as tu
import utils as u

MODELS = [
    ("altra_repo/gab_loocv_c1_1", "best_model.pth.tar"),
    ("altra_repo/gab_loocv_c1_2", "best_model.pth.tar"),
    ("altra_repo/gab_loocv_c2_1", "best_model.pth.tar"),
    ("altra_repo/gab_loocv_c2_2", "best_model.pth.tar"),
    ("gab_h_c1_1_50", "checkpoint_phase_4_best.pth.tar"),
    ("gab_h_c1_2_50", "checkpoint_phase_4_best.pth.tar"),
    ("gab_h_c2_1_50", "checkpoint_phase_4_best.pth.tar"),
    ("gab_h_c2_2_50", "checkpoint_phase_4_best.pth.tar"),
]

ARGS = SimpleNamespace(
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


def load_model(model_path, device):
    """Load a multi-model GCN checkpoint and return (gcn, ARGS)."""
    gcn = egcn_h.EGCN(ARGS, activation=nn.RReLU(), device=device)
    checkpoint = torch.load(model_path, map_location=device)
    gcn.load_state_dict(checkpoint["gcn_dict"])
    gcn = gcn.to(device)
    gcn.eval()
    return gcn


def compute_gcn_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list):
    """Compute embeddings from a multi-model GCN and return NumPy array."""
    with torch.inference_mode():
        nodes_embs = gcn(hist_adj_list, hist_ndFeats_list, mask_list)
    return nodes_embs.detach().cpu().numpy()


def main():
    """Entry point for multi-model GCN embedding analysis."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("MULTI-MODEL GCN EMBEDDINGS COMPARISON")
    print("=" * 60)

    print("\nLoading real dataset...")
    real_args = SimpleNamespace(
        gab_args={"folder": str(project_root / "data/") + "/", "feats_per_node": 768}
    )
    dataset = GabDataset(real_args)
    max_snapshot = int(dataset.max_time.item())

    adj = tu.get_sp_adj(edges=dataset.edges, time=max_snapshot, time_window=None)
    features = dataset.get_temporal_node_features(max_snapshot)
    num_real_nodes = dataset.num_nodes

    adj_norm = tu.normalize_adj(adj, num_real_nodes)
    mask = tu.get_node_mask(adj_norm, num_real_nodes)

    adj_batched = {
        "idx": adj_norm["idx"].unsqueeze(0),
        "vals": adj_norm["vals"].unsqueeze(0),
    }
    adj_tensor = u.sparse_prepare_tensor(adj_batched, [num_real_nodes]).to(device)

    hist_adj_list = [adj_tensor]
    hist_ndFeats_list = [features.to(device)]
    mask_list = [mask.to(device)]

    print(f"Real nodes: {num_real_nodes}")

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
    print(f"Synthetic users: {len(synth_features)}")

    feat_matrix = np.vstack([features.numpy(), synth_features])
    total_nodes = num_real_nodes + len(synth_features)

    real_idx = adj["idx"]
    real_vals = adj["vals"]
    synth_ids = torch.arange(num_real_nodes, total_nodes)
    self_loops = torch.stack([synth_ids, synth_ids], dim=1)
    all_idx = torch.cat([real_idx, self_loops], dim=0)
    all_vals = torch.cat([real_vals, torch.ones(len(synth_ids), dtype=real_vals.dtype)])

    hybrid_adj = {"idx": all_idx, "vals": all_vals}
    adj_norm_h = tu.normalize_adj(hybrid_adj, total_nodes)
    mask_h = tu.get_node_mask(adj_norm_h, total_nodes)

    adj_batched_h = {
        "idx": adj_norm_h["idx"].unsqueeze(0),
        "vals": adj_norm_h["vals"].unsqueeze(0),
    }
    adj_tensor_h = u.sparse_prepare_tensor(adj_batched_h, [total_nodes]).to(device)

    hist_adj_list_h = [adj_tensor_h]
    hist_ndFeats_list_h = [torch.tensor(feat_matrix, dtype=torch.float).to(device)]
    mask_list_h = [mask_h.to(device)]

    all_results = {}

    for model_folder, model_file in MODELS:
        model_name = model_folder.split("/")[-1]
        model_path = project_root / "log" / model_folder / model_file

        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}")

        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            continue

        gcn = load_model(str(model_path), device)

        print("  Computing GCN embeddings for real nodes...")
        real_gcn_embs = compute_gcn_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list)

        print("  Computing GCN embeddings for hybrid graph...")
        all_gcn_embs = compute_gcn_embeddings(
            gcn, hist_adj_list_h, hist_ndFeats_list_h, mask_list_h
        )

        synth_gcn_embs = all_gcn_embs[num_real_nodes:]

        n_real = min(1000, len(real_gcn_embs))
        n_synth = min(1000, len(synth_gcn_embs))

        real_idx_sample = np.random.choice(len(real_gcn_embs), n_real, replace=False)
        synth_idx_sample = np.random.choice(len(synth_gcn_embs), n_synth, replace=False)

        real_sample = real_gcn_embs[real_idx_sample]
        synth_sample = synth_gcn_embs[synth_idx_sample]

        real_cos = 1 - pdist(real_sample, metric="cosine")
        synth_cos = 1 - pdist(synth_sample, metric="cosine")
        cross_cos = 1 - cdist(real_sample, synth_sample, metric="cosine")

        real_var = real_gcn_embs.var(axis=0).mean()
        synth_var = synth_gcn_embs.var(axis=0).mean()

        results = {
            "real_cosine_similarity": {
                "mean": float(real_cos.mean()),
                "std": float(real_cos.std()),
                "min": float(real_cos.min()),
                "max": float(real_cos.max()),
            },
            "synthetic_cosine_similarity": {
                "mean": float(synth_cos.mean()),
                "std": float(synth_cos.std()),
                "min": float(synth_cos.min()),
                "max": float(synth_cos.max()),
            },
            "cross_group_cosine_similarity": {
                "mean": float(cross_cos.mean()),
                "std": float(cross_cos.std()),
            },
            "real_variance": float(real_var),
            "synthetic_variance": float(synth_var),
            "variance_ratio": float(real_var / synth_var) if synth_var > 0 else float("inf"),
        }

        all_results[model_name] = results

        print(f"\n  Real cosine sim: mean={real_cos.mean():.4f}, std={real_cos.std():.4f}")
        print(f"  Synth cosine sim: mean={synth_cos.mean():.4f}, std={synth_cos.std():.4f}")
        print(f"  Cross-group sim: mean={cross_cos.mean():.4f}")
        print(f"  Variance ratio: {real_var / synth_var:.2f}x")

        del gcn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_dir / "multi_model_gcn_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved results to {output_dir / 'multi_model_gcn_comparison.json'}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'Real Sim':>10} {'Synth Sim':>10} {'Ratio':>10}")
    print("-" * 60)
    for model_name, results in all_results.items():
        print(
            f"{model_name:<25} {results['real_cosine_similarity']['mean']:>10.4f} "
            f"{results['synthetic_cosine_similarity']['mean']:>10.4f} "
            f"{results['variance_ratio']:>10.1f}x"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = list(all_results.keys())
    real_means = [all_results[m]["real_cosine_similarity"]["mean"] for m in model_names]
    synth_means = [all_results[m]["synthetic_cosine_similarity"]["mean"] for m in model_names]
    variance_ratios = [all_results[m]["variance_ratio"] for m in model_names]
    cross_means = [all_results[m]["cross_group_cosine_similarity"]["mean"] for m in model_names]

    ax = axes[0, 0]
    x = range(len(model_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], real_means, width, label="Real", alpha=0.7)
    ax.bar([i + width / 2 for i in x], synth_means, width, label="Synthetic", alpha=0.7)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Cosine Similarity by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("gab_", "").replace("_50", "") for m in model_names], rotation=45, ha="right"
    )
    ax.legend()

    ax = axes[0, 1]
    ax.bar(model_names, variance_ratios, alpha=0.7, color="orange")
    ax.set_ylabel("Variance Ratio (Real/Synth)")
    ax.set_title("Variance Ratio by Model")
    ax.set_xticklabels(
        [m.replace("gab_", "").replace("_50", "") for m in model_names], rotation=45, ha="right"
    )

    ax = axes[1, 0]
    real_stds = [all_results[m]["real_cosine_similarity"]["std"] for m in model_names]
    synth_stds = [all_results[m]["synthetic_cosine_similarity"]["std"] for m in model_names]
    ax.bar([i - width / 2 for i in x], real_stds, width, label="Real", alpha=0.7)
    ax.bar([i + width / 2 for i in x], synth_stds, width, label="Synthetic", alpha=0.7)
    ax.set_ylabel("Std Dev of Cosine Similarity")
    ax.set_title("Similarity Std Dev by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("gab_", "").replace("_50", "") for m in model_names], rotation=45, ha="right"
    )
    ax.legend()

    ax = axes[1, 1]
    ax.bar(model_names, cross_means, alpha=0.7, color="green")
    ax.set_ylabel("Mean Cross-Group Similarity")
    ax.set_title("Real-Synth Similarity by Model")
    ax.set_xticklabels(
        [m.replace("gab_", "").replace("_50", "") for m in model_names], rotation=45, ha="right"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "multi_model_gcn_comparison.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'multi_model_gcn_comparison.png'}")


if __name__ == "__main__":
    main()
