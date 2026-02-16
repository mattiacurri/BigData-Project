"""Visualize embeddings using PCA and t-SNE."""

import json
from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


def compute_user_embeddings_from_posts(
    post_embeddings: dict, posts_df, user_col: str = "user_id", post_col: str = "post_id"
) -> dict:
    """Aggregate post embeddings into per-user embeddings (mean across posts)."""
    user_embs = {}
    for user_id, group in posts_df.groupby(user_col):
        embeddings = []
        for pid in group[post_col].values:
            pid_int = int(pid)
            if pid_int in post_embeddings:
                emb = post_embeddings[pid_int]
                embeddings.append(emb.numpy() if hasattr(emb, "numpy") else emb)
        if embeddings:
            user_embs[user_id] = np.array(embeddings).mean(axis=0)
    return user_embs


def visualize_user_embeddings(
    project_root: Path, output_dir: Path, real_emb: dict, synth_emb: dict
):
    """Create PCA and t-SNE visualizations for user embeddings and save plots."""
    from sklearn.decomposition import PCA

    print("\n" + "=" * 60)
    print("USER EMBEDDINGS VISUALIZATION")
    print("=" * 60)

    print("\nLoading real posts metadata...")
    time_periods = ["2016-2021", "2022", "2023", "2024", "Jan-Jul 25", "Jul 25"]
    real_posts_dfs = []
    for period in time_periods:
        posts_file = project_root / "data/raw" / period / "posts_current_snapshot.csv"
        if posts_file.exists():
            df = pd.read_csv(posts_file)
            df["user_id"] = df["account_id"]
            df["post_id"] = df["id"]
            real_posts_dfs.append(df[["user_id", "post_id"]])
    real_posts_df = (
        pd.concat(real_posts_dfs, ignore_index=True) if real_posts_dfs else pd.DataFrame()
    )

    print("\nLoading synthetic posts metadata...")
    batch_files = [
        project_root / "data/raw/batch1_synthetic.csv",
        project_root / "data/raw/batch2_synthetic.csv",
        project_root / "data/raw/batch3_synthetic.csv",
    ]
    synth_posts_dfs = []
    for bf in batch_files:
        if bf.exists():
            df = pd.read_csv(bf)
            if "user_id" in df.columns and "post_id" in df.columns:
                synth_posts_dfs.append(df[["user_id", "post_id"]])
    synth_posts_df = (
        pd.concat(synth_posts_dfs, ignore_index=True) if synth_posts_dfs else pd.DataFrame()
    )

    print(f"Real posts: {len(real_posts_df)}")
    print(f"Synthetic posts: {len(synth_posts_df)}")

    print("\nComputing user embeddings (mean of post embeddings)...")
    real_user_embs = compute_user_embeddings_from_posts(
        real_emb, real_posts_df, "user_id", "post_id"
    )
    synth_user_embs = compute_user_embeddings_from_posts(
        synth_emb, synth_posts_df, "user_id", "post_id"
    )

    print(f"Real users with embeddings: {len(real_user_embs)}")
    print(f"Synthetic users with embeddings: {len(synth_user_embs)}")

    real_user_arr = np.array(list(real_user_embs.values()))
    synth_user_arr = np.array(list(synth_user_embs.values()))

    n_real_sample = min(3000, len(real_user_arr))
    n_synth_sample = min(3000, len(synth_user_arr))

    real_idx = np.random.choice(len(real_user_arr), n_real_sample, replace=False)
    synth_idx = np.random.choice(len(synth_user_arr), n_synth_sample, replace=False)

    real_sample = real_user_arr[real_idx]
    synth_sample = synth_user_arr[synth_idx]

    combined = np.vstack([real_sample, synth_sample])

    print("\nPCA for user embeddings...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    real_pca = pca_result[:n_real_sample]
    synth_pca = pca_result[n_real_sample:]

    print("\nt-SNE for user embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
    tsne_result = tsne.fit_transform(combined)

    real_tsne = tsne_result[:n_real_sample]
    synth_tsne = tsne_result[n_real_sample:]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=10, label="Real Users", c="blue")
    plt.scatter(
        synth_pca[:, 0], synth_pca[:, 1], alpha=0.3, s=10, label="Synthetic Users", c="red"
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA: User Embeddings (Mean of Posts)")
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.3, s=10, label="Real Users", c="blue")
    plt.scatter(
        synth_tsne[:, 0], synth_tsne[:, 1], alpha=0.3, s=10, label="Synthetic Users", c="red"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: User Embeddings (Mean of Posts)")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "user_embeddings_visualization.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_dir / 'user_embeddings_visualization.png'}")


def main():
    """Run embedding visualizations (PCA and t-SNE) for posts and users and save outputs."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EMBEDDINGS VISUALIZATION (PCA, t-SNE)")
    print("=" * 60)

    print("\nLoading BERT embeddings...")
    with open(project_root / "data/raw/bert_features_real_posts.pkl", "rb") as f:
        real_emb = pickle.load(f)
    with open(project_root / "data/raw/bert_features_synthetic.pkl", "rb") as f:
        synth_emb = pickle.load(f)

    real_embs = []
    for pid, emb in real_emb.items():
        real_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
    real_embs = np.array(real_embs)

    synth_embs = []
    for pid, emb in synth_emb.items():
        synth_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
    synth_embs = np.array(synth_embs)

    print(f"Real embeddings: {real_embs.shape}")
    print(f"Synthetic embeddings: {synth_embs.shape}")

    n_real_sample = min(5000, len(real_embs))
    n_synth_sample = min(5000, len(synth_embs))

    real_idx = np.random.choice(len(real_embs), n_real_sample, replace=False)
    synth_idx = np.random.choice(len(synth_embs), n_synth_sample, replace=False)

    real_sample = real_embs[real_idx]
    synth_sample = synth_embs[synth_idx]

    combined = np.vstack([real_sample, synth_sample])

    print(f"Combined sample: {combined.shape}")

    stats = {
        "n_real_sampled": n_real_sample,
        "n_synth_sampled": n_synth_sample,
    }

    print("\n" + "=" * 60)
    print("PCA VISUALIZATION")
    print("=" * 60)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained: {sum(pca.explained_variance_ratio_):.2%}")

    stats["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained": float(sum(pca.explained_variance_ratio_)),
    }

    real_pca = pca_result[:n_real_sample]
    synth_pca = pca_result[n_real_sample:]

    centroid_real = real_pca.mean(axis=0)
    centroid_synth = synth_pca.mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_real - centroid_synth)

    print(f"Centroid distance: {centroid_distance:.4f}")

    stats["pca"]["centroid_distance"] = float(centroid_distance)

    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION")
    print("=" * 60)

    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
    tsne_result = tsne.fit_transform(combined)

    real_tsne = tsne_result[:n_real_sample]
    synth_tsne = tsne_result[n_real_sample:]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=5, label="Real", c="blue")
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.3, s=5, label="Synthetic", c="red")
    plt.scatter(
        centroid_real[0],
        centroid_real[1],
        c="darkblue",
        s=100,
        marker="x",
        linewidths=3,
        label="Real centroid",
    )
    plt.scatter(
        centroid_synth[0],
        centroid_synth[1],
        c="darkred",
        s=100,
        marker="x",
        linewidths=3,
        label="Synth centroid",
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA: BERT Embeddings")
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.3, s=5, label="Real", c="blue")
    plt.scatter(synth_tsne[:, 0], synth_tsne[:, 1], alpha=0.3, s=5, label="Synthetic", c="red")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: BERT Embeddings")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_visualization_bert.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_dir / 'embeddings_visualization_bert.png'}")

    visualize_user_embeddings(project_root, output_dir, real_emb, synth_emb)

    gcn_real_path = output_dir / "real_gcn_embeddings.npy"
    gcn_synth_path = output_dir / "synth_gcn_embeddings.npy"

    if gcn_real_path.exists() and gcn_synth_path.exists():
        print("\n" + "=" * 60)
        print("GCN EMBEDDINGS VISUALIZATION")
        print("=" * 60)

        real_gcn = np.load(gcn_real_path)
        synth_gcn = np.load(gcn_synth_path)

        print(f"Real GCN embeddings: {real_gcn.shape}")
        print(f"Synthetic GCN embeddings: {synth_gcn.shape}")

        n_real = min(5000, len(real_gcn))
        n_synth = min(1000, len(synth_gcn))

        real_idx = np.random.choice(len(real_gcn), n_real, replace=False)
        synth_idx = np.random.choice(len(synth_gcn), n_synth, replace=False)

        real_gcn_sample = real_gcn[real_idx]
        synth_gcn_sample = synth_gcn[synth_idx]

        combined_gcn = np.vstack([real_gcn_sample, synth_gcn_sample])

        stats["gcn"] = {
            "n_real_sampled": n_real,
            "n_synth_sampled": n_synth,
        }

        print("\nPCA for GCN embeddings...")
        pca_gcn = PCA(n_components=2)
        pca_gcn_result = pca_gcn.fit_transform(combined_gcn)

        print(f"Explained variance ratio: {pca_gcn.explained_variance_ratio_}")
        print(f"Total explained: {sum(pca_gcn.explained_variance_ratio_):.2%}")

        stats["gcn"]["pca"] = {
            "explained_variance_ratio": pca_gcn.explained_variance_ratio_.tolist(),
            "total_explained": float(sum(pca_gcn.explained_variance_ratio_)),
        }

        real_pca_gcn = pca_gcn_result[:n_real]
        synth_pca_gcn = pca_gcn_result[n_real:]

        centroid_real_gcn = real_pca_gcn.mean(axis=0)
        centroid_synth_gcn = synth_pca_gcn.mean(axis=0)
        centroid_distance_gcn = np.linalg.norm(centroid_real_gcn - centroid_synth_gcn)

        print(f"Centroid distance: {centroid_distance_gcn:.4f}")
        stats["gcn"]["pca"]["centroid_distance"] = float(centroid_distance_gcn)

        print("\nt-SNE for GCN embeddings...")
        tsne_gcn = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
        tsne_gcn_result = tsne_gcn.fit_transform(combined_gcn)

        real_tsne_gcn = tsne_gcn_result[:n_real]
        synth_tsne_gcn = tsne_gcn_result[n_real:]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(real_pca_gcn[:, 0], real_pca_gcn[:, 1], alpha=0.3, s=5, label="Real", c="blue")
        plt.scatter(
            synth_pca_gcn[:, 0], synth_pca_gcn[:, 1], alpha=0.3, s=5, label="Synthetic", c="red"
        )
        plt.scatter(
            centroid_real_gcn[0],
            centroid_real_gcn[1],
            c="darkblue",
            s=100,
            marker="x",
            linewidths=3,
        )
        plt.scatter(
            centroid_synth_gcn[0],
            centroid_synth_gcn[1],
            c="darkred",
            s=100,
            marker="x",
            linewidths=3,
        )
        plt.xlabel(f"PC1 ({pca_gcn.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca_gcn.explained_variance_ratio_[1]:.1%})")
        plt.title("PCA: GCN Embeddings")
        plt.legend(fontsize=8)

        plt.subplot(1, 2, 2)
        plt.scatter(
            real_tsne_gcn[:, 0], real_tsne_gcn[:, 1], alpha=0.3, s=5, label="Real", c="blue"
        )
        plt.scatter(
            synth_tsne_gcn[:, 0], synth_tsne_gcn[:, 1], alpha=0.3, s=5, label="Synthetic", c="red"
        )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE: GCN Embeddings")
        plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "embeddings_visualization_gcn.png", dpi=150)
        plt.close()
        print(f"Saved plot to {output_dir / 'embeddings_visualization_gcn.png'}")

    with open(output_dir / "visualization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {output_dir / 'visualization_stats.json'}")


if __name__ == "__main__":
    main()
