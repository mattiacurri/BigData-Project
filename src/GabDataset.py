"""Dataset loader and processor.

Loads and preprocesses temporal graph data from social network interactions.
"""

from pathlib import Path
import pickle
from typing import Dict, Optional

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
        args.gab_args = u.Namespace(args.gab_args)

        # Setup for edge loading
        self.folder_path = Path(args.gab_args.folder)
        self.raw_folder = self.folder_path / "raw"
        self.time_periods = ["2016-2021", "2022", "2023", "2024", "Jan-Jul 25", "Jul 25"]
        self.loaded_snapshots = set()  # Track which snapshots have been loaded

        self.ecols = u.Namespace({"FromNodeId": 0, "ToNodeId": 1, "Weight": 2, "TimeStep": 3})

        # Load ALL snapshots to initialize the dataset structure
        print("Loading all snapshots for dataset initialization...")
        edges_df = self._load_all_snapshots()

        edges = torch.tensor(edges_df[["X", "Y", "Label", "Snapshot"]].values, dtype=torch.float32)
        edges = self.make_contigous_node_ids(edges)

        # Save args for later use
        self.args_gab = args.gab_args

        # Compute nodes per snapshot for incremental learning
        timesteps = u.aggregate_by_time(edges[:, self.ecols.TimeStep], args.gab_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:, self.ecols.TimeStep] = timesteps

        # Track which nodes appear in each snapshot (for incremental learning)
        self._compute_nodes_per_snapshot(edges)

        # Total number of nodes across ALL snapshots (for backward compatibility)
        num_nodes = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]].unique().size(0)

        # edges +1: positive
        # edges -1: negative
        # separate
        # majority voting for the final label
        edges[:, self.ecols.Weight] = (
            self.cluster_negs_and_positives()
        )  # edges[:, self.ecols.Weight])

        # separate classes
        sp_indices = (
            edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId, self.ecols.TimeStep]].long().t()
        )
        sp_values = edges[:, self.ecols.Weight]

        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse_coo_tensor(
            neg_sp_indices,
            neg_sp_values,
            size=(num_nodes, num_nodes, int(self.max_time.item()) + 1),
        ).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse_coo_tensor(
            pos_sp_indices,
            pos_sp_values,
            size=(num_nodes, num_nodes, int(self.max_time.item()) + 1),
        ).coalesce()

        # scale positive class to separate after adding
        # we ensure that if it's a positive label it will be >0 after substraction
        pos_sp_edges *= 1000

        # we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        # separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        # We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        # creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)

        # # the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        # vals = pos_vals + neg_vals

        # Unweighted graph: all edges have weight 1
        vals = torch.ones(vals.size(0), dtype=torch.float32)

        self.edges = {"idx": indices_labels, "vals": vals}
        self.num_nodes = num_nodes
        self.num_classes = 2

        # Create post-to-snapshot mapping
        self.post_to_snapshot, self.post_to_user = self._create_post_mappings(args.gab_args)

        # Create user ID mapping
        unique_users = set(edges_df["X"].unique()) | set(edges_df["Y"].unique())
        original_user_ids = sorted(list(unique_users))
        self.user_id_to_node_idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
        self.node_idx_to_user_id = {
            idx: user_id for user_id, idx in self.user_id_to_node_idx.items()
        }

        # Embeddings file path (loaded on-demand in get_temporal_node_features)
        self._embeddings_file = Path(args.gab_args.folder) / "raw" / "bert_features_real_posts.pkl"
        self.feats_per_node = 768  # BERT embedding dimension

        print(f"Features per node: {self.feats_per_node}")
        print(f"Number of nodes (total across all snapshots): {self.num_nodes}")
        print(f"Nodes per snapshot: {[len(nodes) for nodes in self.nodes_per_snapshot.values()]}")
        print(f"Post-to-snapshot mappings: {len(self.post_to_snapshot)}")

    def _initialize_node_id_map(self, edges: torch.Tensor) -> None:
        """Initialize node ID mapping from initial edges.

        Args:
            edges: Initial edge tensor with mapped node IDs.
        """
        # Get unique original node IDs from edges
        from_nodes = edges[:, self.ecols.FromNodeId].long()
        to_nodes = edges[:, self.ecols.ToNodeId].long()
        all_nodes = torch.cat([from_nodes, to_nodes]).unique()

        # Create mapping
        for idx, node_id in enumerate(all_nodes.tolist()):
            self._node_id_map[node_id] = idx

        self._next_node_id = len(all_nodes)
        print(f"Initialized node ID map with {self._next_node_id} nodes")

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
                    "X": edges[:, 0].numpy(),
                    "Y": edges[:, 1].numpy(),
                    "Snapshot": snapshot_idx,
                    "Label": 1,
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
            DataFrame with columns ['X', 'Y', 'Snapshot', 'Label'].
        """
        if snapshot_idx in self.loaded_snapshots:
            print(f"Snapshot {snapshot_idx} already loaded, skipping...")
            return pd.DataFrame(columns=["X", "Y", "Snapshot", "Label"])

        if snapshot_idx >= len(self.time_periods):
            raise ValueError(
                f"Snapshot index {snapshot_idx} out of range [0, {len(self.time_periods)})"
            )

        period = self.time_periods[snapshot_idx]
        period_folder = self.raw_folder / period

        if not period_folder.exists():
            raise FileNotFoundError(f"Folder {period_folder} does not exist")

        edges_file = period_folder / "social_network.edg"
        if not edges_file.exists():
            raise FileNotFoundError(f"File {edges_file} does not exist")

        print(f"Loading edges from snapshot {snapshot_idx} ({period})...")
        edges = self._load_edg_file(edges_file)

        snapshot_df = pd.DataFrame(
            {
                "X": edges[:, 0].numpy(),
                "Y": edges[:, 1].numpy(),
                "Snapshot": snapshot_idx,
                "Label": 1,
            }
        )

        self.loaded_snapshots.add(snapshot_idx)
        print(f"  Loaded {len(snapshot_df)} edges from snapshot {snapshot_idx}")

        return snapshot_df

    def add_snapshot_edges(self, snapshot_idx: int) -> None:
        """Add edges from a new snapshot to the dataset.

        This method loads edges for the specified snapshot and REBUILDS the dataset
        with all loaded snapshots. This is simpler than trying to merge incremental edges.

        Args:
            snapshot_idx: Index of the snapshot to add.
        """
        if snapshot_idx in self.loaded_snapshots:
            print(f"Snapshot {snapshot_idx} already loaded")
            return

        # Load the new snapshot edges (marks it as loaded)
        new_edges_df = self._load_snapshot_edges(snapshot_idx)

        if len(new_edges_df) == 0:
            return

        # Now rebuild the dataset with ALL loaded snapshots
        # This is simpler than trying to incrementally merge
        print(f"Rebuilding dataset with snapshots: {sorted(self.loaded_snapshots)}")
        self._rebuild_dataset_with_loaded_snapshots()

    def _rebuild_dataset_with_loaded_snapshots(self) -> None:
        """Rebuild the dataset structure using all currently loaded snapshots.

        This method re-processes all loaded edges and reconstructs the sparse
        edge structure, node mappings, and other dataset attributes.
        """
        # Load all edges from loaded snapshots
        all_edges_list = []
        for snap_idx in sorted(self.loaded_snapshots):
            snap_df = self._get_loaded_snapshot_edges(snap_idx)
            all_edges_list.append(snap_df)

        edges_df = pd.concat(all_edges_list, ignore_index=True)

        # Convert to tensor
        edges = torch.tensor(edges_df[["X", "Y", "Label", "Snapshot"]].values, dtype=torch.float32)
        edges = self.make_contigous_node_ids(edges)

        # Process edges (same as original __init__)
        timesteps = u.aggregate_by_time(edges[:, self.ecols.TimeStep], self.args_gab.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:, self.ecols.TimeStep] = timesteps

        # Update nodes per snapshot
        self._compute_nodes_per_snapshot(edges)

        # Update total number of nodes
        num_nodes = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]].unique().size(0)

        # Process labels
        edges[:, self.ecols.Weight] = self.cluster_negs_and_positives()

        # Create sparse representation (same as __init__)
        sp_indices = (
            edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId, self.ecols.TimeStep]].long().t()
        )
        sp_values = edges[:, self.ecols.Weight]

        neg_mask = sp_values == -1
        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse_coo_tensor(
            neg_sp_indices,
            neg_sp_values,
            size=(num_nodes, num_nodes, int(self.max_time.item()) + 1),
        ).coalesce()

        pos_mask = sp_values == 1
        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]
        pos_sp_edges = torch.sparse_coo_tensor(
            pos_sp_indices,
            pos_sp_values,
            size=(num_nodes, num_nodes, int(self.max_time.item()) + 1),
        ).coalesce()

        pos_sp_edges *= 1000
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        vals = pos_vals - neg_vals

        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)

        vals = torch.ones(vals.size(0), dtype=torch.float32)

        # Update dataset attributes
        self.edges = {"idx": indices_labels, "vals": vals}
        self.num_nodes = num_nodes

        # Update user ID mappings
        unique_users = set(edges_df["X"].unique()) | set(edges_df["Y"].unique())
        original_user_ids = sorted(list(unique_users))
        self.user_id_to_node_idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
        self.node_idx_to_user_id = {
            idx: user_id for user_id, idx in self.user_id_to_node_idx.items()
        }

        print(f"Rebuilt dataset: {self.num_nodes} nodes, {self.edges['idx'].size(0)} edges")

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
                "X": edges[:, 0].numpy(),
                "Y": edges[:, 1].numpy(),
                "Snapshot": snapshot_idx,
                "Label": 1,
            }
        )
        return snapshot_df

    def _compute_nodes_per_snapshot(self, edges: torch.Tensor) -> None:
        """Compute which nodes appear up to each snapshot (cumulative).

        This enables truly incremental learning where early snapshots
        only have features for nodes that exist at that point in time.

        Args:
            edges: Edge tensor with FromNodeId, ToNodeId, Weight, TimeStep columns.
        """
        self.nodes_per_snapshot: Dict[int, set] = {}

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
            self.nodes_per_snapshot[snapshot] = all_nodes_so_far.copy()

        print(f"\nNodes per snapshot (cumulative):")
        for snapshot in sorted(self.nodes_per_snapshot.keys()):
            print(f"  Snapshot {snapshot}: {len(self.nodes_per_snapshot[snapshot])} nodes")

    def get_num_nodes_at_snapshot(self, snapshot: int) -> int:
        """Get the number of nodes that exist up to the given snapshot.

        Args:
            snapshot: Snapshot index.

        Returns:
            Number of nodes that have appeared up to this snapshot.
        """
        # if snapshot in self.nodes_per_snapshot:
        return len(self.nodes_per_snapshot[snapshot])
        # Fallback to total if snapshot not found
        # return self.num_nodes

    def get_node_indices_at_snapshot(self, snapshot: int) -> torch.Tensor:
        """Get indices of nodes that exist up to the given snapshot.

        Args:
            snapshot: Snapshot index.

        Returns:
            Tensor of node indices that exist up to this snapshot.
        """
        # if snapshot in self.nodes_per_snapshot:
        return torch.tensor(sorted(list(self.nodes_per_snapshot[snapshot])), dtype=torch.long)
        # # Fallback to all nodes
        # return torch.arange(self.num_nodes, dtype=torch.long)

    def cluster_negs_and_positives(self):  # , ratings):
        """Convert ratings to binary labels (+1/-1).

        Args:
            ratings: Rating values (positive or negative).

        Returns:
            Binary tensor with +1 for positive, -1 for negative/zero ratings.
        """
        # pos_indices = ratings > 0
        # neg_indices = ratings <= 0
        # ratings[pos_indices] = 1
        # ratings[neg_indices] = -1
        # return ratings
        return 1

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

        Args:
            max_snapshot: Maximum snapshot index (inclusive) to use for features.

        Returns:
            Tensor of shape [self.num_nodes, embedding_dim] with temporal features.
            Nodes without features or that don't exist yet have zero vectors.
        """
        # Get nodes that exist at this snapshot for filtering
        nodes_at_snapshot = set(self.get_node_indices_at_snapshot(max_snapshot).tolist())

        # Load embeddings directly from disk (don't keep in memory)
        # This trades CPU for RAM
        if not self._embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self._embeddings_file}")

        print(f"Loading embeddings for snapshot {max_snapshot}...")
        import pickle

        with open(self._embeddings_file, "rb") as f:
            all_embeddings = pickle.load(f)

        # Initialize feature matrix for ALL nodes
        node_features = torch.zeros(self.num_nodes, self.feats_per_node, dtype=torch.float32)

        # Group posts by user, filtering by snapshot
        user_embeddings: Dict[int, list] = {}

        for post_id, embedding in all_embeddings.items():
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

        # Free memory from raw embeddings
        del all_embeddings

        # Compute average embedding for each user
        users_with_features = 0
        for user_id, embeddings in user_embeddings.items():
            node_idx = self.user_id_to_node_idx[user_id]
            user_posts = torch.stack(embeddings)
            avg_embedding = user_posts.mean(dim=0)
            node_features[node_idx] = avg_embedding
            users_with_features += 1

        num_nodes_at_snapshot = len(nodes_at_snapshot)

        print(
            f"Computed temporal features for snapshot {max_snapshot}: "
            f"{users_with_features}/{num_nodes_at_snapshot} active nodes with features "
            f"({100 * users_with_features / num_nodes_at_snapshot:.1f}% coverage), "
            f"tensor shape: {node_features.shape}"
        )
        return node_features

    def prepare_node_feats(self, node_feats):
        """Prepare node features for model input.

        Args:
            node_feats: Raw node features (can be temporal).

        Returns:
            Processed node features.
        """
        # node_feats is already a tensor with shape [num_nodes, embedding_dim]
        # Just ensure it's float for model compatibility
        return node_feats.float()

    def edges_to_sp_dict(self, edges):
        """Convert edge list to sparse dictionary format.

        Args:
            edges: Edge list with source, target, weight, time.

        Returns:
            Dict with 'idx' (edge indices) and 'vals' (weights).
        """
        idx = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId, self.ecols.TimeStep]]

        vals = edges[:, self.ecols.Weight]
        return {"idx": idx, "vals": vals}

    def get_num_nodes(self, edges):
        """Get the number of unique nodes in the edge list.

        Args:
            edges: Edge list.

        Returns:
            int: Number of nodes.
        """
        all_ids = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_and_save_edges(self, _args):
        """DEPRECATED: Use incremental loading instead.

        This method is kept for backward compatibility but is no longer used.
        The dataset now loads edges incrementally using _load_snapshot_edges()
        and add_snapshot_edges().

        Args:
            _args: Configuration with folder path.

        Returns:
            DataFrame with columns ['X', 'Y', 'Snapshot', 'Label'].
        """
        raise NotImplementedError(
            "load_and_save_edges is deprecated. Use _load_snapshot_edges() and add_snapshot_edges() instead."
        )

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
        edges_result[:, self.ecols.Weight] = edges[:, self.ecols.Weight]
        edges_result[:, self.ecols.TimeStep] = edges[:, self.ecols.TimeStep]

        return edges_result
