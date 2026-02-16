"""Compare real vs synthetic embeddings."""

import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist


def compute_user_embeddings(
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


def main():
    """Compare real and synthetic embeddings at post and user levels and save metrics."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(project_root / "data/raw/bert_features_real_posts.pkl", "rb") as f:
        real_emb = pickle.load(f)
    with open(project_root / "data/raw/bert_features_synthetic.pkl", "rb") as f:
        synth_emb = pickle.load(f)

    print("=" * 60)
    print("REAL vs SYNTHETIC EMBEDDINGS COMPARISON")
    print("=" * 60)

    real_embs = []
    for pid, emb in real_emb.items():
        real_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
    real_embs = np.array(real_embs)

    synth_embs = []
    for pid, emb in synth_emb.items():
        synth_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
    synth_embs = np.array(synth_embs)

    print(f"\nReal posts: {len(real_embs)}")
    print(f"Synthetic posts: {len(synth_embs)}")

    n_samples = 1000
    real_sample = real_embs[np.random.choice(len(real_embs), n_samples, replace=False)]
    synth_sample = synth_embs[
        np.random.choice(len(synth_embs), min(n_samples, len(synth_embs)), replace=False)
    ]

    comparison = {}

    print(f"\n--- Cosine Similarity (POSTS) ---")
    real_cos = 1 - pdist(real_sample, metric="cosine")
    synth_cos = 1 - pdist(synth_sample, metric="cosine")
    print(f"Real-Real: mean={real_cos.mean():.4f}, std={real_cos.std():.4f}")
    print(f"Synth-Synth: mean={synth_cos.mean():.4f}, std={synth_cos.std():.4f}")
    comparison["cosine_similarity_posts"] = {
        "real_real": {"mean": float(real_cos.mean()), "std": float(real_cos.std())},
        "synth_synth": {"mean": float(synth_cos.mean()), "std": float(synth_cos.std())},
    }

    print(f"\n--- Cross-group Similarity (POSTS) ---")
    cross_cos = 1 - cdist(real_sample, synth_sample, metric="cosine")
    print(f"Real-Synth: mean={cross_cos.mean():.4f}, std={cross_cos.std():.4f}")
    comparison["cosine_similarity_posts"]["real_synth"] = {
        "mean": float(cross_cos.mean()),
        "std": float(cross_cos.std()),
    }

    print(f"\n--- Euclidean Distance ---")
    real_euc = pdist(real_sample, metric="euclidean")
    synth_euc = pdist(synth_sample, metric="euclidean")
    cross_euc = cdist(real_sample, synth_sample, metric="euclidean")
    print(f"Real-Real: mean={real_euc.mean():.4f}, std={real_euc.std():.4f}")
    print(f"Synth-Synth: mean={synth_euc.mean():.4f}, std={synth_euc.std():.4f}")
    print(f"Real-Synth: mean={cross_euc.mean():.4f}, std={cross_euc.std():.4f}")
    comparison["euclidean"] = {
        "real_real": {"mean": float(real_euc.mean()), "std": float(real_euc.std())},
        "synth_synth": {"mean": float(synth_euc.mean()), "std": float(synth_euc.std())},
        "real_synth": {"mean": float(cross_euc.mean()), "std": float(cross_euc.std())},
    }

    print(f"\n--- Variance Comparison (POSTS) ---")
    real_var = real_embs.var(axis=0)
    synth_var = synth_embs.var(axis=0)
    combined_var = np.vstack([real_embs, synth_embs]).var(axis=0)

    real_centroid = real_embs.mean(axis=0)
    synth_centroid = synth_embs.mean(axis=0)
    centroid_distance = np.linalg.norm(real_centroid - synth_centroid)

    print(f"Real within-group variance (mean per dim): {real_var.mean():.6f}")
    print(f"Synth within-group variance (mean per dim): {synth_var.mean():.6f}")
    print(f"Combined variance (mean per dim): {combined_var.mean():.6f}")
    print(f"Within-group variance ratio (real/synth): {real_var.mean() / synth_var.mean():.2f}x")
    print(f"Cross-group centroid distance: {centroid_distance:.4f}")

    comparison["variance_posts"] = {
        "real_within_group_variance": float(real_var.mean()),
        "synth_within_group_variance": float(synth_var.mean()),
        "combined_variance": float(combined_var.mean()),
        "within_group_ratio": float(real_var.mean() / synth_var.mean()),
        "cross_group_centroid_distance": float(centroid_distance),
    }

    print("\n" + "=" * 60)
    print("USER-LEVEL VARIANCE ANALYSIS")
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

    print("Loading synthetic posts metadata...")
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

    print("Computing user embeddings...")
    real_user_embs = compute_user_embeddings(real_emb, real_posts_df)
    synth_user_embs = compute_user_embeddings(synth_emb, synth_posts_df)

    print(f"Real users: {len(real_user_embs)}")
    print(f"Synthetic users: {len(synth_user_embs)}")

    real_user_arr = np.array(list(real_user_embs.values()))
    synth_user_arr = np.array(list(synth_user_embs.values()))

    print(f"\n--- Variance Comparison (USERS) ---")
    real_user_var = real_user_arr.var(axis=0)
    synth_user_var = synth_user_arr.var(axis=0)
    combined_user_arr = np.vstack([real_user_arr, synth_user_arr])
    combined_user_var = combined_user_arr.var(axis=0)

    real_user_centroid = real_user_arr.mean(axis=0)
    synth_user_centroid = synth_user_arr.mean(axis=0)
    centroid_user_distance = np.linalg.norm(real_user_centroid - synth_user_centroid)

    print(f"Real user variance (mean per dim): {real_user_var.mean():.6f}")
    print(f"Synth user variance (mean per dim): {synth_user_var.mean():.6f}")
    print(f"Combined user variance (mean per dim): {combined_user_var.mean():.6f}")
    print(
        f"Within-group variance ratio (real/synth): {real_user_var.mean() / synth_user_var.mean():.2f}x"
    )
    print(f"Cross-group centroid distance: {centroid_user_distance:.4f}")

    comparison["variance_users"] = {
        "real_within_group_variance": float(real_user_var.mean()),
        "synth_within_group_variance": float(synth_user_var.mean()),
        "combined_variance": float(combined_user_var.mean()),
        "within_group_ratio": float(real_user_var.mean() / synth_user_var.mean()),
        "cross_group_centroid_distance": float(centroid_user_distance),
        "n_real_users": len(real_user_embs),
        "n_synth_users": len(synth_user_embs),
    }

    n_user_samples = min(500, len(real_user_arr), len(synth_user_arr))
    real_user_sample = real_user_arr[
        np.random.choice(len(real_user_arr), n_user_samples, replace=False)
    ]
    synth_user_sample = synth_user_arr[
        np.random.choice(len(synth_user_arr), n_user_samples, replace=False)
    ]

    print(f"\n--- Cosine Similarity (USERS) ---")
    real_user_cos = 1 - pdist(real_user_sample, metric="cosine")
    synth_user_cos = 1 - pdist(synth_user_sample, metric="cosine")
    cross_user_cos = 1 - cdist(real_user_sample, synth_user_sample, metric="cosine")
    print(f"Real-Real Users: mean={real_user_cos.mean():.4f}, std={real_user_cos.std():.4f}")
    print(f"Synth-Synth Users: mean={synth_user_cos.mean():.4f}, std={synth_user_cos.std():.4f}")
    print(f"Real-Synth Users: mean={cross_user_cos.mean():.4f}, std={cross_user_cos.std():.4f}")
    comparison["cosine_similarity_users"] = {
        "real_real": {"mean": float(real_user_cos.mean()), "std": float(real_user_cos.std())},
        "synth_synth": {"mean": float(synth_user_cos.mean()), "std": float(synth_user_cos.std())},
        "real_synth": {"mean": float(cross_user_cos.mean()), "std": float(cross_user_cos.std())},
    }

    print(f"\n--- Summary Statistics ---")
    print(f"Real: mean={real_embs.mean():.4f}, std={real_embs.std():.4f}")
    print(f"Synth: mean={synth_embs.mean():.4f}, std={synth_embs.std():.4f}")
    comparison["global_stats"] = {
        "real": {"mean": float(real_embs.mean()), "std": float(real_embs.std())},
        "synth": {"mean": float(synth_embs.mean()), "std": float(synth_embs.std())},
    }

    with open(output_dir / "embedding_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison to {output_dir / 'embedding_comparison.json'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(real_cos, bins=50, alpha=0.6, label="Real-Real", density=True)
    ax.hist(synth_cos, bins=50, alpha=0.6, label="Synth-Synth", density=True)
    ax.hist(cross_cos.flatten(), bins=50, alpha=0.6, label="Real-Synth", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Distribution")
    ax.legend()

    ax = axes[1]
    ax.hist(real_euc, bins=50, alpha=0.6, label="Real-Real", density=True)
    ax.hist(synth_euc, bins=50, alpha=0.6, label="Synth-Synth", density=True)
    ax.hist(cross_euc.flatten(), bins=50, alpha=0.6, label="Real-Synth", density=True)
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Density")
    ax.set_title("Euclidean Distance Distribution")
    ax.legend()

    ax = axes[2]
    ax.plot(real_var, alpha=0.7, label="Real")
    ax.plot(synth_var, alpha=0.7, label="Synthetic")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Variance")
    ax.set_title("Variance per Dimension (Posts)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "embedding_comparison.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'embedding_comparison.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    x_labels = ["Real Posts", "Synth Posts"]
    y_vals = [real_var.mean(), synth_var.mean()]
    bars = ax.bar(x_labels, y_vals, color=["blue", "orange"])
    ax.set_ylabel("Mean Variance per Dimension")
    ax.set_title(
        f"POST Variance\nReal: {real_var.mean():.6f}, Synth: {synth_var.mean():.6f}\nRatio: {real_var.mean() / synth_var.mean():.2f}x"
    )
    for bar, val in zip(bars, y_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    x_labels = ["Real Users", "Synth Users"]
    y_vals = [real_user_var.mean(), synth_user_var.mean()]
    bars = ax.bar(x_labels, y_vals, color=["blue", "orange"])
    ax.set_ylabel("Mean Variance per Dimension")
    ax.set_title(
        f"USER Variance\nReal: {real_user_var.mean():.6f}, Synth: {synth_user_var.mean():.6f}\nRatio: {real_user_var.mean() / synth_user_var.mean():.2f}x"
    )
    for bar, val in zip(bars, y_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[2]
    categories = ["Posts Real", "Posts Synth", "Users Real", "Users Synth"]
    values = [real_var.mean(), synth_var.mean(), real_user_var.mean(), synth_user_var.mean()]
    colors = ["blue", "orange", "blue", "orange"]
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Mean Variance per Dimension")
    ax.set_title("Variance Comparison: Posts vs Users")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "embedding_variance_summary.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'embedding_variance_summary.png'}")


if __name__ == "__main__":
    main()
