"""Plotting helpers for TSNE visualization of embeddings."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

# Import y-sunflower styling libraries
try:
    import morethemes as mt

    mt.set_theme("minimal")  # Clean, modern theme
    MORETHEMES_AVAILABLE = True
except ImportError:
    MORETHEMES_AVAILABLE = False

# try:
#     from pypalettes import load_cmap

#     PYPALETTES_AVAILABLE = True
# except ImportError:
#     PYPALETTES_AVAILABLE = False

# try:
#     from pyfonts import load_google_font

#     PYFONTS_AVAILABLE = True
# except ImportError:
#     PYFONTS_AVAILABLE = False

# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot t-SNE embeddings from CSV files.")
parser.add_argument("files", nargs="+", help="Paths to the CSV files containing embeddings.")
args = parser.parse_args()


# Apply custom styling if libraries not available
def style_axis(ax, is_3d=False):
    """Apply consistent styling to axes."""
    # Remove top and right spines for cleaner look
    if not is_3d:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    # Style the grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


# Plot from best learned embeddings
files = args.files
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file, header=None)
        embeddings = df.iloc[:, 1:].values  # skip first column orig_ID
        print(f"Loaded {file} with shape {embeddings.shape}")

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=50, max_iter=1000)
        embeddings_3d = tsne_3d.fit_transform(embeddings)

        # Create figure with better proportions
        fig = plt.figure(figsize=(18, 7), facecolor="white")

        # Single color for all points
        point_color = "#2E5C8A"

        # 2D subplot
        ax1 = fig.add_subplot(121)
        ax1.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=point_color,
            alpha=0.4,
            s=3,
            edgecolors="none",
        )
        style_axis(ax1)
        ax1.set_title(
            "t-SNE Node Embeddings - 2D Projection",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax1.set_xlabel("t-SNE Component 1", fontsize=11)
        ax1.set_ylabel("t-SNE Component 2", fontsize=11)

        # 3D subplot
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            embeddings_3d[:, 0],
            embeddings_3d[:, 1],
            embeddings_3d[:, 2],
            c=point_color,
            alpha=0.3,
            s=4,
            edgecolors="none",
        )
        ax2.set_title(
            "t-SNE Node Embeddings - 3D Projection",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax2.set_xlabel("t-SNE 1", fontsize=10, labelpad=5)
        ax2.set_ylabel("t-SNE 2", fontsize=10, labelpad=5)
        ax2.set_zlabel("t-SNE 3", fontsize=10, labelpad=5)

        # Style 3D plot
        ax2.grid(True, alpha=0.2)
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor("gray")
        ax2.yaxis.pane.set_edgecolor("gray")
        ax2.zaxis.pane.set_edgecolor("gray")
        ax2.xaxis.pane.set_alpha(0.1)
        ax2.yaxis.pane.set_alpha(0.1)
        ax2.zaxis.pane.set_alpha(0.1)

        # Adjust layout
        plt.tight_layout(pad=3.0)

        # Save with high quality
        os.makedirs("tsne", exist_ok=True)
        output_file = (
            f"tsne/tsne_learned_{os.path.basename(file).replace('.csv', '')}_combined.png"
        )
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved plot to {output_file}")
        plt.close()
    else:
        print(f"File {file} not found.")
