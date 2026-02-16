"""Analyze real BERT embeddings distribution and variance."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist


def main():
    """Analyze real embeddings distribution and save stats and plots."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(project_root / "data/raw/bert_features_real_posts.pkl", "rb") as f:
        real_emb = pickle.load(f)

    print("=" * 60)
    print("REAL BERT EMBEDDINGS ANALYSIS")
    print("=" * 60)

    embs = []
    for pid, emb in real_emb.items():
        embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
    embs = np.array(embs)

    print(f"\nNumber of posts: {len(embs)}")
    print(f"Embedding dim: {embs.shape[1]}")

    stats = {
        "num_posts": len(embs),
        "embedding_dim": embs.shape[1],
        "global_mean": float(embs.mean()),
        "global_std": float(embs.std()),
        "global_min": float(embs.min()),
        "global_max": float(embs.max()),
    }

    print(f"\n--- Global Statistics ---")
    print(f"Mean: {stats['global_mean']:.6f}")
    print(f"Std: {stats['global_std']:.6f}")
    print(f"Min: {stats['global_min']:.6f}")
    print(f"Max: {stats['global_max']:.6f}")

    var_per_dim = embs.var(axis=0)
    print(f"\n--- Variance per Dimension ---")
    print(f"Mean var: {var_per_dim.mean():.6f}")
    print(f"Min var: {var_per_dim.min():.6f}")
    print(f"Max var: {var_per_dim.max():.6f}")
    stats["var_per_dim_mean"] = float(var_per_dim.mean())
    stats["var_per_dim_min"] = float(var_per_dim.min())
    stats["var_per_dim_max"] = float(var_per_dim.max())

    n_samples = min(2000, len(embs))
    sample_indices = np.random.choice(len(embs), n_samples, replace=False)
    sample = embs[sample_indices]

    print(f"\n--- Distance Analysis (sample of {n_samples}) ---")
    euclidean_dist = pdist(sample, metric="euclidean")
    print(f"Euclidean mean: {euclidean_dist.mean():.4f}")
    print(f"Euclidean std: {euclidean_dist.std():.4f}")
    print(f"Euclidean min: {euclidean_dist.min():.4f}")
    print(f"Euclidean max: {euclidean_dist.max():.4f}")
    stats["euclidean_mean"] = float(euclidean_dist.mean())
    stats["euclidean_std"] = float(euclidean_dist.std())

    cos_sim = 1 - pdist(sample, metric="cosine")
    print(f"\n--- Cosine Similarity ---")
    print(f"Mean: {cos_sim.mean():.4f}")
    print(f"Std: {cos_sim.std():.4f}")
    print(f"Min: {cos_sim.min():.4f}")
    print(f"Max: {cos_sim.max():.4f}")
    stats["cosine_sim_mean"] = float(cos_sim.mean())
    stats["cosine_sim_std"] = float(cos_sim.std())

    print(f"\n--- Cosine Similarity Distribution ---")
    cosine_dist = {}
    for low, high in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        count = np.sum((cos_sim >= low) & (cos_sim < high))
        pct = count / len(cos_sim) * 100
        print(f"[{low:.2f}-{high:.2f}): {count:>8,} ({pct:>5.1f}%)")
        cosine_dist[f"{low:.2f}-{high:.2f}"] = {"count": int(count), "pct": float(pct)}
    stats["cosine_similarity_distribution"] = cosine_dist

    print(f"\n--- Percentiles ---")
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = float(np.percentile(cos_sim, p))
        print(f"P{p}: {val:.4f}")
        percentiles[f"p{p}"] = val
    stats["cosine_similarity_percentiles"] = percentiles

    import json

    with open(output_dir / "real_embeddings_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {output_dir / 'real_embeddings_stats.json'}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(cos_sim, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Real Embeddings: Cosine Similarity Distribution")
    plt.axvline(cos_sim.mean(), color="red", linestyle="--", label=f"Mean: {cos_sim.mean():.3f}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(euclidean_dist, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.title("Real Embeddings: Euclidean Distance Distribution")
    plt.axvline(
        euclidean_dist.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {euclidean_dist.mean():.3f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "real_embeddings_distribution.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'real_embeddings_distribution.png'}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(768), var_per_dim)
    plt.xlabel("Dimension")
    plt.ylabel("Variance")
    plt.title("Variance per Dimension - Real Embeddings")
    plt.tight_layout()
    plt.savefig(output_dir / "real_variance_per_dim.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'real_variance_per_dim.png'}")


if __name__ == "__main__":
    main()
