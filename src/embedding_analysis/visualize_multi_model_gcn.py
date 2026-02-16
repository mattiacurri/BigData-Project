"""Visualize GCN embeddings for each model with PCA and t-SNE."""

import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import nn

warnings.filterwarnings("ignore")

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
    """Load and return a multi-model GCN instance and its ARGS."""
    gcn = egcn_h.EGCN(ARGS, activation=nn.RReLU(), device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gcn.load_state_dict(checkpoint["gcn_dict"])
    gcn = gcn.to(device)
    gcn.eval()
    return gcn


def compute_gcn_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list):
    """Compute and return node embeddings from `gcn` as NumPy array."""
    with torch.inference_mode():
        nodes_embs = gcn(hist_adj_list, hist_ndFeats_list, mask_list)
    return nodes_embs.detach().cpu().numpy()


def main():
    """Visualize embeddings produced by multi-model GCNs and save plots."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results" / "gcn_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("MULTI-MODEL GCN VISUALIZATION")
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

    all_visualization_stats = {}

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

        print("  Computing GCN embeddings...")
        all_gcn_embs = compute_gcn_embeddings(
            gcn, hist_adj_list_h, hist_ndFeats_list_h, mask_list_h
        )

        real_gcn_embs = all_gcn_embs[:num_real_nodes]
        synth_gcn_embs = all_gcn_embs[num_real_nodes:]

        n_real = min(2000, len(real_gcn_embs))
        n_synth = min(1000, len(synth_gcn_embs))

        real_idx_s = np.random.choice(len(real_gcn_embs), n_real, replace=False)
        synth_idx_s = np.random.choice(len(synth_gcn_embs), n_synth, replace=False)

        real_sample = real_gcn_embs[real_idx_s]
        synth_sample = synth_gcn_embs[synth_idx_s]

        combined = np.vstack([real_sample, synth_sample])
        labels = np.array([0] * n_real + [1] * n_synth)

        print("  Computing PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)

        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
        tsne_result = tsne.fit_transform(combined)

        real_pca = pca_result[:n_real]
        synth_pca = pca_result[n_real:]
        real_tsne = tsne_result[:n_real]
        synth_tsne = tsne_result[n_real:]

        centroid_real_pca = real_pca.mean(axis=0)
        centroid_synth_pca = synth_pca.mean(axis=0)
        centroid_distance = np.linalg.norm(centroid_real_pca - centroid_synth_pca)

        vis_stats = {
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "pca_total_explained": float(sum(pca.explained_variance_ratio_)),
            "pca_centroid_distance": float(centroid_distance),
        }
        all_visualization_stats[model_name] = vis_stats

        print(f"  PCA explained: {sum(pca.explained_variance_ratio_):.2%}")
        print(f"  Centroid distance: {centroid_distance:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=5, label="Real", c="blue")
        ax.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, s=10, label="Synthetic", c="red")
        ax.scatter(
            centroid_real_pca[0],
            centroid_real_pca[1],
            c="darkblue",
            s=100,
            marker="x",
            linewidths=3,
        )
        ax.scatter(
            centroid_synth_pca[0],
            centroid_synth_pca[1],
            c="darkred",
            s=100,
            marker="x",
            linewidths=3,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"PCA: {model_name}")
        ax.legend()

        ax = axes[1]
        ax.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.3, s=5, label="Real", c="blue")
        ax.scatter(synth_tsne[:, 0], synth_tsne[:, 1], alpha=0.5, s=10, label="Synthetic", c="red")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE: {model_name}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_gcn_visualization.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / f'{model_name}_gcn_visualization.png'}")

        del gcn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Creating combined overview plot...")
    print("=" * 60)

    n_models = len(MODELS)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))

    for i, (model_folder, _) in enumerate(MODELS):
        model_name = model_folder.split("/")[-1]
        img_path = output_dir / f"{model_name}_gcn_visualization.png"

        if img_path.exists():
            img = plt.imread(img_path)
            axes[0, i].imshow(img[:, : img.shape[1] // 2, :])
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{model_name}\nPCA", fontsize=8)

            axes[1, i].imshow(img[:, img.shape[1] // 2 :, :])
            axes[1, i].axis("off")
            axes[1, i].set_title("t-SNE", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "all_models_gcn_overview.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'all_models_gcn_overview.png'}")

    with open(output_dir / "visualization_stats.json", "w") as f:
        json.dump(all_visualization_stats, f, indent=2)
    print(f"Saved: {output_dir / 'visualization_stats.json'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'PCA Explained':>15} {'Centroid Dist':>15}")
    print("-" * 60)
    for model_name, stats in all_visualization_stats.items():
        print(
            f"{model_name:<25} {stats['pca_total_explained']:>14.2%} {stats['pca_centroid_distance']:>15.4f}"
        )


if __name__ == "__main__":
    main()
