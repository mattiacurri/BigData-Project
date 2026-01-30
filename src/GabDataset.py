"""Dataset loader and processor.

Loads and preprocesses temporal graph data from social network interactions.
"""

from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Dict

import pandas as pd
import torch
from tqdm import tqdm

import utils as u


class GabDataset:
    """Social network dataset with temporal edge information."""

    def __init__(self, args):
        """Initialize and load the social network dataset.

        Args:
            args: Configuration namespace with dataset parameters.

        Raises:
            AssertionError: If task is not 'link_pred' or 'edge_cls'.
        """
        assert args.task in ["link_pred", "edge_cls"], (
            "dataset only implements link_pred or edge_cls"
        )
        args.gab_args = SimpleNamespace(**args.gab_args)

        # Setup for edge loading
        self.folder_path = Path(args.gab_args.folder)
        self.raw_folder = self.folder_path / "raw"
        self.time_periods = ["2016-2021", "2022", "2023", "2024", "Jan-Jul 25", "Jul 25"]
        self.loaded_snapshots = set()  # Track which snapshots have been loaded

        self.ecols = SimpleNamespace(FromNodeId=0, ToNodeId=1, TimeStep=2)

        edges_df = self._load_all_snapshots()

        edges = torch.tensor(
            edges_df[["FromNodeId", "ToNodeId", "TimeStep"]].values, dtype=torch.float32
        )
        edges = self.make_contigous_node_ids(edges)

        # Save args for later use
        self.args_gab = args.gab_args

        # Compute nodes per snapshot for incremental learning
        # timesteps = u.aggregate_by_time(edges[:, self.ecols.TimeStep], args.gab_args.aggr_time)
        timesteps = edges[:, self.ecols.TimeStep]  # Usa timestamp originali senza aggregazione
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        # edges[:, self.ecols.TimeStep] = timesteps  # Non necessario se non modifichiamo

        # Track which nodes appear in each snapshot (for incremental learning)
        self._compute_nodes_per_snapshot(
            edges
        )  # mask the nodes that do not exist yet at each snapshot

        # Create sparse representation for edges
        sp_indices = (
            edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId, self.ecols.TimeStep]].long().t()
        )

        # Create labels: all existing edges are positive (1)
        new_vals = torch.ones(sp_indices.size(1), dtype=torch.long)
        # [from, to, time, label]
        indices_labels = torch.cat([sp_indices.t(), new_vals.view(-1, 1)], dim=1)

        # All edges have value 1
        vals = torch.ones(sp_indices.size(1), dtype=torch.float32)

        self.edges = {"idx": indices_labels, "vals": vals}

        # Total number of nodes across all snapshots
        # !!! For now, we are excluding users without connections across all snapshots
        self.num_nodes = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]].unique().size(0)

        # Create post-to-snapshot mapping
        self.post_to_snapshot, self.post_to_user = self._create_post_mappings(args.gab_args)

        # Create user ID mapping
        unique_users = set(edges_df["FromNodeId"].unique()) | set(edges_df["ToNodeId"].unique())
        original_user_ids = sorted(list(unique_users))
        self.user_id_to_node_idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
        self.node_idx_to_user_id = {
            idx: user_id for user_id, idx in self.user_id_to_node_idx.items()
        }

        # Embeddings file path (loaded on-demand in get_temporal_node_features)
        self._embeddings_file = Path(args.gab_args.folder) / "raw" / "bert_features_real_posts.pkl"

        with open(self._embeddings_file, "rb") as f:
            self.all_embeddings = pickle.load(f)

        self.feats_per_node = 768  # BERT embedding dimension

        # Cache for temporal node features (snapshot -> tensor)
        self._temporal_features_cache: Dict[int, torch.Tensor] = {}

        print(f"Features per node: {self.feats_per_node}")
        print(f"Number of nodes (total across all snapshots): {self.num_nodes}")
        print(
            f"Nodes per snapshot: {[len(nodes) for nodes in self.cumulative_nodes_per_snapshot.values()]}"
        )
        print(f"Post-to-snapshot mappings: {len(self.post_to_snapshot)}")

    def _load_all_snapshots(self) -> pd.DataFrame:
        """Load edges from all snapshots.

        Returns:
            DataFrame with all edges from all snapshots.
        """
        all_edges = []

        for snapshot_idx, period in enumerate(self.time_periods):
            period_folder = self.raw_folder / period

            if not period_folder.exists():
                print(f"Warning: Folder {period_folder} does not exist, skipping...")
                continue

            edges_file = period_folder / "social_network.edg"
            if not edges_file.exists():
                print(f"Warning: {edges_file} does not exist, skipping...")
                continue

            print(f"Loading snapshot {snapshot_idx} ({period})...")
            edges = self._load_edg_file(edges_file)

            snapshot_df = pd.DataFrame(
                {
                    "FromNodeId": edges[:, 0].numpy(),
                    "ToNodeId": edges[:, 1].numpy(),
                    "TimeStep": snapshot_idx,
                }
            )

            all_edges.append(snapshot_df)
            self.loaded_snapshots.add(snapshot_idx)
            print(f"  Loaded {len(snapshot_df)} edges from snapshot {snapshot_idx}")

        if not all_edges:
            raise ValueError("No edge files found in the data folder")

        edges_df = pd.concat(all_edges, ignore_index=True)
        print(f"\nTotal edges loaded: {len(edges_df)} from {len(all_edges)} snapshots")

        return edges_df

    def _load_snapshot_edges(self, snapshot_idx: int) -> pd.DataFrame:
        """Load edges for a specific snapshot.

        Args:
            snapshot_idx: Index of the snapshot to load.

        Returns:
            DataFrame with columns ['FromNodeId', 'ToNodeId', 'TimeStep'].
        """
        if snapshot_idx in self.loaded_snapshots:
            print(f"Snapshot {snapshot_idx} already loaded, skipping...")
            return pd.DataFrame(columns=["FromNodeId", "ToNodeId", "TimeStep"])

        if snapshot_idx >= len(self.time_periods):
            raise ValueError(
                f"Snapshot index {snapshot_idx} out of range [0, {len(self.time_periods)})"
            )

        period = self.time_periods[snapshot_idx]
        period_folder = self.raw_folder / period

        edges_file = period_folder / "social_network.edg"

        print(f"Loading edges from snapshot {snapshot_idx} ({period})...")
        edges = self._load_edg_file(edges_file)

        snapshot_df = pd.DataFrame(
            {
                "FromNodeId": edges[:, 0].numpy(),
                "ToNodeId": edges[:, 1].numpy(),
                "TimeStep": snapshot_idx,
            }
        )

        self.loaded_snapshots.add(snapshot_idx)
        print(f"  Loaded {len(snapshot_df)} edges from snapshot {snapshot_idx}")

        return snapshot_df

    def _get_loaded_snapshot_edges(self, snapshot_idx: int) -> pd.DataFrame:
        """Reload edges from a previously loaded snapshot.

        Args:
            snapshot_idx: Index of the snapshot.

        Returns:
            DataFrame with edges from this snapshot.
        """
        if snapshot_idx >= len(self.time_periods):
            raise ValueError(f"Snapshot index {snapshot_idx} out of range")

        period = self.time_periods[snapshot_idx]
        period_folder = self.raw_folder / period
        edges_file = period_folder / "social_network.edg"

        edges = self._load_edg_file(edges_file)
        snapshot_df = pd.DataFrame(
            {
                "FromNodeId": edges[:, 0].numpy(),
                "ToNodeId": edges[:, 1].numpy(),
                "TimeStep": snapshot_idx,
            }
        )

        print(f"Reloaded {len(snapshot_df)} edges from snapshot {snapshot_idx}")
        return snapshot_df

    def _compute_nodes_per_snapshot(self, edges: torch.Tensor) -> None:
        """Compute which nodes appear up to each snapshot (cumulative).

        This enables truly incremental learning where early snapshots
        only have features for nodes that exist at that point in time.

        Args:
            edges: Edge tensor with FromNodeId, ToNodeId, TimeStep columns.
        """
        self.cumulative_nodes_per_snapshot: Dict[int, set] = {}

        min_time = int(self.min_time.item())
        max_time = int(self.max_time.item())

        # For each snapshot, track all nodes that appear up to that point
        all_nodes_so_far = set()

        for snapshot in range(min_time, max_time + 1):
            # Get edges in this snapshot
            mask = edges[:, self.ecols.TimeStep] == snapshot
            snapshot_edges = edges[mask]

            # Get unique nodes in this snapshot
            if snapshot_edges.size(0) > 0:
                from_nodes = snapshot_edges[:, self.ecols.FromNodeId].long().tolist()
                to_nodes = snapshot_edges[:, self.ecols.ToNodeId].long().tolist()
                snapshot_nodes = set(from_nodes + to_nodes)
                all_nodes_so_far.update(snapshot_nodes)

            # Store cumulative nodes (all nodes seen up to this snapshot)
            self.cumulative_nodes_per_snapshot[snapshot] = all_nodes_so_far.copy()

        print(f"\nNodes per snapshot (cumulative):")
        for snapshot in sorted(self.cumulative_nodes_per_snapshot.keys()):
            print(
                f"  Snapshot {snapshot}: {len(self.cumulative_nodes_per_snapshot[snapshot])} nodes"
            )

    def get_node_indices_at_snapshot(self, snapshot: int) -> torch.Tensor:
        """Get indices of nodes that exist up to the given snapshot.

        Args:
            snapshot: Snapshot index.

        Returns:
            Tensor of node indices that exist up to this snapshot.
        """
        return torch.tensor(
            sorted(list(self.cumulative_nodes_per_snapshot[snapshot])), dtype=torch.long
        )

    def _create_post_mappings(self, gab_args) -> tuple[Dict[int, int], Dict[int, int]]:
        """Create mappings from post_id to snapshot and post_id to user_id.

        Args:
            gab_args: Configuration with folder path.

        Returns:
            Tuple of (post_to_snapshot, post_to_user) dictionaries.
        """
        folder_path = Path(gab_args.folder)
        raw_folder = folder_path / "raw"
        time_periods = ["2016-2021", "2022", "2023", "2024", "Jan-Jul 25", "Jul 25"]

        post_to_snapshot: Dict[int, int] = {}
        post_to_user: Dict[int, int] = {}

        print("Creating post-to-snapshot and post-to-user mappings...")
        for snapshot_idx, period in enumerate(tqdm(time_periods, desc="Processing periods")):
            period_folder = raw_folder / period
            posts_file = period_folder / "posts_current_snapshot.csv"

            if posts_file.exists():
                posts_df = pd.read_csv(posts_file)
                for _, row in posts_df.iterrows():
                    post_id = int(row["id"])
                    user_id = int(row["account_id"])
                    post_to_snapshot[post_id] = snapshot_idx
                    post_to_user[post_id] = user_id

        print(f"  Created {len(post_to_snapshot)} post mappings")
        return post_to_snapshot, post_to_user

    def get_temporal_node_features(self, max_snapshot: int) -> torch.Tensor:
        """Compute node features using only posts up to the given snapshot.

        This allows the model to see how user features evolve over time,
        using only historical information available at each snapshot.

        IMPORTANT: Returns features for ALL nodes in the dataset (self.num_nodes),
        with zeros for nodes that don't exist yet or don't have features.
        This is necessary because node IDs are contiguous across all snapshots.

        Uses caching to avoid recomputing features for the same snapshot multiple times.

        Args:
            max_snapshot: Maximum snapshot index (inclusive) to use for features.

        Returns:
            Tensor of shape [self.num_nodes, embedding_dim] with temporal features.
            Nodes without features or that don't exist yet have zero vectors.
        """
        # Check cache first
        if max_snapshot in self._temporal_features_cache:
            return self._temporal_features_cache[max_snapshot]

        # ??? For now we are doing in this way
        # train on snapshot 0 -> mean(emb_0)
        # test on snapshot 1 -> mean(emb_0)
        # train on snapshot 1 -> mean(emb_0, emb_1)
        # test on snapshot 2 -> mean(emb_0, emb_1)

        # Get nodes that exist at this snapshot for filtering
        nodes_at_snapshot = set(self.get_node_indices_at_snapshot(max_snapshot).tolist())

        # Initialize feature matrix for ALL nodes
        node_features = torch.zeros(self.num_nodes, self.feats_per_node, dtype=torch.float32)

        # Group posts by user, filtering by snapshot
        user_embeddings: Dict[int, list] = {}

        for post_id, embedding in self.all_embeddings.items():
            # Check if post exists in our mappings and is within snapshot range
            if post_id in self.post_to_snapshot and post_id in self.post_to_user:
                post_snapshot = self.post_to_snapshot[post_id]

                # Only include posts up to max_snapshot
                if post_snapshot <= max_snapshot:
                    user_id = self.post_to_user[post_id]

                    # Only include users that appear in our graph
                    if user_id in self.user_id_to_node_idx:
                        node_idx = self.user_id_to_node_idx[user_id]
                        # Only include if this node exists at this snapshot
                        if node_idx in nodes_at_snapshot:
                            if user_id not in user_embeddings:
                                user_embeddings[user_id] = []
                            user_embeddings[user_id].append(embedding)

        # Compute average embedding for each user
        users_with_features = 0
        for user_id, embeddings in user_embeddings.items():
            node_idx = self.user_id_to_node_idx[user_id]
            user_posts = torch.stack(embeddings)
            avg_embedding = user_posts.mean(dim=0)
            node_features[node_idx] = avg_embedding
            users_with_features += 1

        # Cache the computed features
        self._temporal_features_cache[max_snapshot] = node_features

        return node_features

    def _load_edg_file(self, file_path):
        """Load edges from .edg file (tab-separated source-target pairs).

        Args:
            file_path: Path to the .edg file.

        Returns:
            Tensor of shape (num_edges, 2) with source and target node IDs.
        """
        edges = []
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    edges.append([int(parts[0]), int(parts[1])])
        return torch.tensor(edges, dtype=torch.long)

    def make_contigous_node_ids(self, edges):
        """Remap node IDs to be contiguous (0 to num_nodes-1).

        Args:
            edges: Edge list with original node IDs.

        Returns:
            Edge list with remapped node IDs.
        """
        # Extract node columns
        node_cols = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]].long()
        _, new_node_ids = node_cols.unique(return_inverse=True)

        # Reshape to match original shape (num_edges, 2)
        new_node_ids = new_node_ids.reshape(-1, 2)

        # Rebuild edges tensor with correct dtypes
        edges_result = torch.zeros_like(edges)
        edges_result[:, self.ecols.FromNodeId] = new_node_ids[:, 0].float()
        edges_result[:, self.ecols.ToNodeId] = new_node_ids[:, 1].float()
        edges_result[:, self.ecols.TimeStep] = edges[:, self.ecols.TimeStep]

        return edges_result
