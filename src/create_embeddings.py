"""Script to generate and save user embeddings and adjacency matrices per snapshot.

Extracts temporal node features (user embeddings) from the GabDataset and saves them
along with the corresponding adjacency matrices for each snapshot.
"""

import argparse
from pathlib import Path
import pickle
from typing import Dict, Tuple

import torch
from tqdm import tqdm
import yaml

from GabDataset import GabDataset
import utils as u


def build_adjacency_matrix(
    edges_dict: Dict[str, torch.Tensor], snapshot: int, num_nodes: int
) -> torch.sparse.FloatTensor:
    """Build sparse adjacency matrix for a specific snapshot.

    Args:
        edges_dict: Dictionary with 'idx' (edge indices with labels) and 'vals' (weights).
        snapshot: Snapshot index to build matrix for.
        num_nodes: Total number of nodes in the graph.

    Returns:
        Sparse adjacency matrix of shape [num_nodes, num_nodes].
    """
    # Extract edges for this snapshot
    edge_indices = edges_dict["idx"]  # Shape: [num_edges, 4] (from, to, time, label)
    edge_weights = edges_dict["vals"]  # Shape: [num_edges]

    # Filter edges for current snapshot
    snapshot_mask = edge_indices[:, 2] == snapshot
    snapshot_edge_indices = edge_indices[snapshot_mask]
    snapshot_edge_weights = edge_weights[snapshot_mask]

    if snapshot_edge_indices.size(0) == 0:
        print(f"  Warning: No edges found for snapshot {snapshot}")
        # Return empty sparse matrix
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(num_nodes, num_nodes),
        )

    # Extract source and target nodes (columns 0 and 1)
    edges_2d = snapshot_edge_indices[:, :2].long().t()  # Shape: [2, num_edges]

    # Create sparse adjacency matrix
    adj_matrix = torch.sparse_coo_tensor(
        edges_2d, snapshot_edge_weights, size=(num_nodes, num_nodes)
    ).coalesce()

    return adj_matrix


def create_and_save_embeddings(
    dataset: GabDataset, output_dir: Path, save_format: str = "pt"
) -> None:
    """Create user embeddings and adjacency matrices for all snapshots and save to disk.

    Args:
        dataset: Loaded GabDataset instance.
        output_dir: Directory to save the embeddings and adjacency matrices.
        save_format: Format to save ('pt' for PyTorch, 'pkl' for pickle).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    min_snapshot = int(dataset.min_time.item())
    max_snapshot = int(dataset.max_time.item())

    print(
        f"\nCreating embeddings and adjacency matrices for snapshots {min_snapshot} to {max_snapshot}..."
    )
    print(f"Output directory: {output_dir}")
    print(f"Save format: {save_format}")
    print(f"Total number of nodes: {dataset.num_nodes}")

    # Store metadata about the saved embeddings
    metadata = {
        "min_snapshot": min_snapshot,
        "max_snapshot": max_snapshot,
        "num_nodes_total": dataset.num_nodes,
        "embedding_dim": dataset.feats_per_node,
        "snapshots": {},
    }

    # Process each snapshot
    for snapshot in tqdm(range(min_snapshot, max_snapshot + 1), desc="Processing snapshots"):
        print(f"\n--- Snapshot {snapshot} ---")

        # Get temporal node features (embeddings) for this snapshot
        # This returns features ONLY for nodes that exist at this snapshot
        node_embeddings = dataset.get_temporal_node_features(snapshot)

        # Get the indices of nodes that exist at this snapshot
        node_indices = dataset.get_node_indices_at_snapshot(snapshot)
        num_nodes_at_snapshot = len(node_indices)

        print(f"  Node embeddings shape: {node_embeddings.shape}")
        print(f"  Nodes at snapshot: {num_nodes_at_snapshot}")

        # Build adjacency matrix for this snapshot
        # Note: adjacency matrix uses full node space, but only nodes in node_indices have edges
        adj_matrix = build_adjacency_matrix(
            dataset.edges,
            snapshot,
            dataset.num_nodes,  # Use total num_nodes for compatibility
        )

        num_edges = adj_matrix._nnz()
        print(f"  Adjacency matrix shape: {adj_matrix.shape}, non-zero: {num_edges}")

        # Save embeddings and adjacency matrix
        snapshot_data = {
            "embeddings": node_embeddings,  # Shape: [num_nodes_at_snapshot, embedding_dim]
            "node_indices": node_indices,  # Indices of nodes that exist at this snapshot
            "adj_matrix": adj_matrix,  # Shape: [num_nodes, num_nodes]
            "snapshot": snapshot,
            "num_nodes": num_nodes_at_snapshot,
            "num_edges": num_edges,
            "embedding_dim": dataset.feats_per_node,
        }

        # Save to file
        if save_format == "pt":
            output_file = output_dir / f"snapshot_{snapshot:02d}.pt"
            torch.save(snapshot_data, output_file)
        elif save_format == "pkl":
            output_file = output_dir / f"snapshot_{snapshot:02d}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(snapshot_data, f)
        else:
            raise ValueError(f"Unknown save format: {save_format}")

        print(f"  Saved to: {output_file}")

        # Update metadata
        metadata["snapshots"][snapshot] = {
            "num_nodes": num_nodes_at_snapshot,
            "num_edges": num_edges,
            "file": str(output_file.name),
        }

        # Free memory after each snapshot
        dataset.clear_temporal_cache()

    # Save metadata
    metadata_file = output_dir / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n\nSaved metadata to: {metadata_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total snapshots processed: {max_snapshot - min_snapshot + 1}")
    print(f"Embedding dimension: {dataset.feats_per_node}")
    print(f"Total nodes across all snapshots: {dataset.num_nodes}")
    print(f"\nSnapshot details:")
    for snapshot in range(min_snapshot, max_snapshot + 1):
        info = metadata["snapshots"][snapshot]
        print(f"  Snapshot {snapshot}: {info['num_nodes']} nodes, {info['num_edges']} edges")
    print("=" * 80)


def load_snapshot_data(snapshot_file: Path) -> Dict:
    """Load embeddings and adjacency matrix for a specific snapshot.

    Args:
        snapshot_file: Path to the snapshot file (.pt or .pkl).

    Returns:
        Dictionary with 'embeddings', 'node_indices', 'adj_matrix', and metadata.
    """
    if snapshot_file.suffix == ".pt":
        return torch.load(snapshot_file)
    elif snapshot_file.suffix == ".pkl":
        with open(snapshot_file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {snapshot_file.suffix}")


def main():
    """Main function to create and save embeddings."""
    parser = argparse.ArgumentParser(
        description="Create and save user embeddings and adjacency matrices per snapshot"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/gab.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/embeddings",
        help="Directory to save embeddings and adjacency matrices",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "pkl"],
        help="Save format (pt=PyTorch, pkl=pickle)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_args = u.Namespace(config)

    # Ensure task is set correctly
    config_args.task = "link_pred"

    print(f"\nInitializing GabDataset...")
    print(f"Data folder: {config_args.gab_args['folder']}")

    # Load dataset
    dataset = GabDataset(config_args)

    # Create and save embeddings
    output_dir = Path(args.output_dir)
    create_and_save_embeddings(dataset, output_dir, save_format=args.format)

    print("\n✓ Done! Embeddings and adjacency matrices saved successfully.")
    print(f"\nTo load a specific snapshot:")
    print(f"  data = torch.load('{output_dir}/snapshot_00.pt')")
    print(f"  embeddings = data['embeddings']  # Shape: [num_nodes, {dataset.feats_per_node}]")
    print(
        f"  adj_matrix = data['adj_matrix']  # Shape: [{dataset.num_nodes}, {dataset.num_nodes}]"
    )
    print(f"  node_indices = data['node_indices']  # Nodes that exist at this snapshot")


if __name__ == "__main__":
    main()
