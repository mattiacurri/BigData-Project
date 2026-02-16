"""Analyze nearest neighbors to understand embedding space structure."""

import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def main():
    """Run nearest neighbors analysis and save statistics and plot."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "scripts/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(project_root / "data/raw/bert_features_real_posts.pkl", "rb") as f:
        real_emb = pickle.load(f)
    with open(project_root / "data/raw/bert_features_synthetic.pkl", "rb") as f:
        synth_emb = pickle.load(f)

    print("=" * 60)
    print("NEAREST NEIGHBORS ANALYSIS")
    print("=" * 60)

    real_embs = []
    real_ids = []
    for pid, emb in real_emb.items():
        real_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
        real_ids.append(pid)
    real_embs = np.array(real_embs)

    synth_embs = []
    synth_ids = []
    for pid, emb in synth_emb.items():
        synth_embs.append(emb.numpy() if hasattr(emb, "numpy") else emb)
        synth_ids.append(pid)
    synth_embs = np.array(synth_embs)

    print(f"\nReal embeddings: {len(real_embs)}")
    print(f"Synthetic embeddings: {len(synth_embs)}")

    stats = {
        "real_count": len(real_embs),
        "synth_count": len(synth_embs),
    }

    n_samples = 500
    real_sample = real_embs[np.random.choice(len(real_embs), n_samples, replace=False)]
    synth_sample = synth_embs[
        np.random.choice(len(synth_embs), min(n_samples, len(synth_embs)), replace=False)
    ]

    combined = np.vstack([real_sample, synth_sample])
    labels = ["real"] * n_samples + ["synth"] * len(synth_sample)

    print(f"\n--- Finding nearest neighbors for sample ---")
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(combined)
    distances, indices = nn.kneighbors(combined)

    real_nn_stats = {"real": 0, "synth": 0}
    synth_nn_stats = {"real": 0, "synth": 0}

    for i in range(n_samples):
        for j in indices[i, 1:]:
            real_nn_stats[labels[j]] += 1

    for i in range(n_samples, len(combined)):
        for j in indices[i, 1:]:
            synth_nn_stats[labels[j]] += 1

    print(f"\n--- Real posts: Nearest neighbor composition ---")
    print(f"Real neighbors: {real_nn_stats['real']}")
    print(f"Synth neighbors: {real_nn_stats['synth']}")
    real_real_ratio = real_nn_stats["real"] / (real_nn_stats["real"] + real_nn_stats["synth"])
    print(f"Real ratio: {real_real_ratio:.2%}")

    print(f"\n--- Synthetic posts: Nearest neighbor composition ---")
    print(f"Real neighbors: {synth_nn_stats['real']}")
    print(f"Synth neighbors: {synth_nn_stats['synth']}")
    synth_real_ratio = synth_nn_stats["real"] / (synth_nn_stats["real"] + synth_nn_stats["synth"])
    print(f"Real ratio: {synth_real_ratio:.2%}")

    stats["real_nn_composition"] = real_nn_stats
    stats["synth_nn_composition"] = synth_nn_stats
    stats["real_real_ratio"] = float(real_real_ratio)
    stats["synth_real_ratio"] = float(synth_real_ratio)

    print(f"\n--- Distance to nearest neighbor ---")
    real_nn_dist = distances[:n_samples, 1]
    synth_nn_dist = distances[n_samples:, 1]

    print(f"Real posts: mean={real_nn_dist.mean():.4f}, std={real_nn_dist.std():.4f}")
    print(f"Synth posts: mean={synth_nn_dist.mean():.4f}, std={synth_nn_dist.std():.4f}")
    stats["real_nn_distance"] = {
        "mean": float(real_nn_dist.mean()),
        "std": float(real_nn_dist.std()),
    }
    stats["synth_nn_distance"] = {
        "mean": float(synth_nn_dist.mean()),
        "std": float(synth_nn_dist.std()),
    }

    with open(output_dir / "nearest_neighbors_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {output_dir / 'nearest_neighbors_stats.json'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.bar(
        ["Real", "Synthetic"],
        [real_nn_stats["real"], real_nn_stats["synth"]],
        color=["blue", "orange"],
    )
    ax.set_ylabel("Count")
    ax.set_title("Real Posts: Nearest Neighbors Composition")

    ax = axes[0, 1]
    ax.bar(
        ["Real", "Synthetic"],
        [synth_nn_stats["real"], synth_nn_stats["synth"]],
        color=["blue", "orange"],
    )
    ax.set_ylabel("Count")
    ax.set_title("Synthetic Posts: Nearest Neighbors Composition")

    ax = axes[1, 0]
    ax.hist(real_nn_dist, bins=30, alpha=0.7, label="Real posts", density=True)
    ax.hist(synth_nn_dist, bins=30, alpha=0.7, label="Synthetic posts", density=True)
    ax.set_xlabel("Distance to Nearest Neighbor (cosine)")
    ax.set_ylabel("Density")
    ax.set_title("Distance to Nearest Neighbor")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(["Real posts", "Synth posts"], [real_real_ratio, synth_real_ratio])
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Nearest Neighbors that are Real")

    plt.tight_layout()
    plt.savefig(output_dir / "nearest_neighbors_analysis.png", dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir / 'nearest_neighbors_analysis.png'}")


if __name__ == "__main__":
    main()
