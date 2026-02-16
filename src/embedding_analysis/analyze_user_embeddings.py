"""Analyze how user embeddings are aggregated from posts."""

import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def main():
    """Analyze user-level aggregated embeddings and save statistics and plots."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(project_root / "data/raw/bert_features_synthetic.pkl", "rb") as f:
        synth_emb = pickle.load(f)

    batch_files = [
        project_root / "data/raw/batch1_synthetic.csv",
        project_root / "data/raw/batch2_synthetic.csv",
        project_root / "data/raw/batch3_synthetic.csv",
    ]

    print("=" * 60)
    print("USER EMBEDDING AGGREGATION ANALYSIS")
    print("=" * 60)

    all_users = {}
    for bf in batch_files:
        df = pd.read_csv(bf)
        for user_id, group in df.groupby("user_id"):
            if user_id not in all_users:
                all_users[user_id] = []
            all_users[user_id].extend(group["post_id"].tolist())

    print(f"\nTotal users: {len(all_users)}")

    posts_per_user = [len(posts) for posts in all_users.values()]
    print(f"\nPosts per user:")
    print(f"  Mean: {np.mean(posts_per_user):.2f}")
    print(f"  Median: {np.median(posts_per_user):.2f}")
    print(f"  Min: {min(posts_per_user)}")
    print(f"  Max: {max(posts_per_user)}")

    stats = {
        "total_users": len(all_users),
        "posts_per_user": {
            "mean": float(np.mean(posts_per_user)),
            "median": float(np.median(posts_per_user)),
            "min": int(min(posts_per_user)),
            "max": int(max(posts_per_user)),
        },
    }

    user_embs = []
    user_vars = []
    user_count = 0
    missing_posts = 0

    for user_id, post_ids in all_users.items():
        embeddings = []
        for pid in post_ids:
            if pid in synth_emb:
                emb = synth_emb[pid]
                embeddings.append(emb.numpy() if hasattr(emb, "numpy") else emb)
            else:
                missing_posts += 1

        if embeddings:
            embs_array = np.array(embeddings)
            user_emb = embs_array.mean(axis=0)
            user_embs.append(user_emb)
            user_vars.append(embs_array.var(axis=0).mean())
            user_count += 1

    print(f"\nUsers with valid embeddings: {user_count}")
    print(f"Missing posts: {missing_posts}")

    user_embs = np.array(user_embs)
    user_vars = np.array(user_vars)

    print(f"\n--- User Embedding Statistics ---")
    print(f"Shape: {user_embs.shape}")
    print(f"Global mean: {user_embs.mean():.6f}")
    print(f"Global std: {user_embs.std():.6f}")
    stats["user_embeddings"] = {
        "shape": list(user_embs.shape),
        "global_mean": float(user_embs.mean()),
        "global_std": float(user_embs.std()),
    }

    n_samples = min(500, len(user_embs))
    sample = user_embs[np.random.choice(len(user_embs), n_samples, replace=False)]
    user_cos_sim = 1 - pdist(sample, metric="cosine")

    print(f"\n--- User Cosine Similarity ---")
    print(f"Mean: {user_cos_sim.mean():.4f}")
    print(f"Std: {user_cos_sim.std():.4f}")
    stats["user_cosine_similarity"] = {
        "mean": float(user_cos_sim.mean()),
        "std": float(user_cos_sim.std()),
    }

    print(f"\n--- Variance WITHIN Users (posts variance) ---")
    print(f"Mean: {user_vars.mean():.6f}")
    print(f"Median: {np.median(user_vars):.6f}")
    print(f"Std: {user_vars.std():.6f}")
    stats["within_user_variance"] = {
        "mean": float(user_vars.mean()),
        "median": float(np.median(user_vars)),
        "std": float(user_vars.std()),
    }

    print(f"\n--- Variance BETWEEN Users ---")
    between_var = user_embs.var(axis=0).mean()
    print(f"Mean: {between_var:.6f}")
    stats["between_user_variance"] = float(between_var)

    print(f"\n--- Ratio Between/Within ---")
    ratio = between_var / user_vars.mean()
    print(f"Ratio: {ratio:.4f}")
    stats["between_within_ratio"] = float(ratio)

    with open(output_dir / "user_embedding_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {output_dir / 'user_embedding_stats.json'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(posts_per_user, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(posts_per_user),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(posts_per_user):.1f}",
    )
    ax.set_xlabel("Posts per User")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Posts per User")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(user_cos_sim, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        user_cos_sim.mean(), color="red", linestyle="--", label=f"Mean: {user_cos_sim.mean():.3f}"
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("User-User Similarity (Aggregated Embeddings)")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(user_vars, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        user_vars.mean(), color="red", linestyle="--", label=f"Mean: {user_vars.mean():.4f}"
    )
    ax.set_xlabel("Within-User Variance")
    ax.set_ylabel("Count")
    ax.set_title("Variance of Posts Within Each User")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(
        ["Within-User", "Between-User"], [user_vars.mean(), between_var], color=["blue", "orange"]
    )
    ax.set_ylabel("Mean Variance")
    ax.set_title("Within vs Between User Variance")

    plt.tight_layout()
    plt.savefig(output_dir / "user_embedding_analysis.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'user_embedding_analysis.png'}")


if __name__ == "__main__":
    main()
