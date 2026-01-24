"""Dataset loader and processor.

Loads and preprocesses temporal graph data from social network interactions.
"""

from datetime import datetime
from functools import lru_cache
import os
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

        # Load edges and save as DataFrame
        edges_df = self.load_and_save_edges(args.gab_args)

        # Convert DataFrame to tensors for compatibility with existing code
        self.ecols = u.Namespace({"FromNodeId": 0, "ToNodeId": 1, "Weight": 2, "TimeStep": 3})

        edges = torch.tensor(edges_df[["X", "Y", "Label", "Snapshot"]].values, dtype=torch.float32)

        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:, self.ecols.TimeStep], args.gab_args.aggr_time)
        # Store as tensors for compatibility with splitter
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:, self.ecols.TimeStep] = timesteps

        edges[:, self.ecols.Weight] = self.cluster_negs_and_positives(edges[:, self.ecols.Weight])

        # add the reversed link to make the graph undirected
        edges = torch.cat(
            [
                edges,
                edges[
                    :,
                    [
                        self.ecols.ToNodeId,
                        self.ecols.FromNodeId,
                        self.ecols.Weight,
                        self.ecols.TimeStep,
                    ],
                ],
            ]
        )

        # separate classes
        sp_indices = (
            edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId, self.ecols.TimeStep]].long().t()
        )
        sp_values = edges[:, self.ecols.Weight]

        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(
            neg_sp_indices,
            neg_sp_values,
            torch.Size([num_nodes, num_nodes, int(self.max_time.item()) + 1]),
        ).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(
            pos_sp_indices,
            pos_sp_values,
            torch.Size([num_nodes, num_nodes, int(self.max_time.item()) + 1]),
        ).coalesce()

        # scale positive class to separate after adding
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

        # the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals

        self.edges = {"idx": indices_labels, "vals": vals}
        self.num_nodes = num_nodes
        self.num_classes = 2

        # Setup lazy loading of node features
        print("\nSetting up temporal node features with lazy loading...")
        self.gab_args = args.gab_args
        self.edges_df = edges_df

        # Create post-to-snapshot mapping
        self.post_to_snapshot, self.post_to_user = self._create_post_mappings(args.gab_args)

        # Create user ID mapping
        unique_users = set(edges_df["X"].unique()) | set(edges_df["Y"].unique())
        original_user_ids = sorted(list(unique_users))
        self.user_id_to_node_idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
        self.node_idx_to_user_id = {
            idx: user_id for user_id, idx in self.user_id_to_node_idx.items()
        }

        # Lazy loading: embeddings loaded on-demand
        self._post_embeddings: Optional[Dict[int, torch.Tensor]] = None
        self._embeddings_file = Path(args.gab_args.folder) / "raw" / "bert_features_real_posts.pkl"
        self._loaded_up_to_snapshot: int = -1  # Track which snapshot we've loaded up to
        self.feats_per_node = 768  # BERT embedding dimension

        # Cache for temporal features (snapshot -> features tensor)
        self._temporal_features_cache: Dict[int, torch.Tensor] = {}

        print(f"Features per node: {self.feats_per_node}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Post-to-snapshot mappings: {len(self.post_to_snapshot)}")
        print(f"Temporal features will be computed on-demand per snapshot")

    def cluster_negs_and_positives(self, ratings):
        """Convert ratings to binary labels (+1/-1).

        Args:
            ratings: Rating values (positive or negative).

        Returns:
            Binary tensor with +1 for positive, -1 for negative/zero ratings.
        """
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings

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

    def _load_embeddings_for_snapshot(self, max_snapshot: int) -> Dict[int, torch.Tensor]:
        """Load post embeddings incrementally up to the given snapshot.

        Only loads embeddings for posts up to max_snapshot, and loads incrementally
        when moving to later snapshots to minimize memory usage.

        Args:
            max_snapshot: Maximum snapshot index (inclusive) to load embeddings for.

        Returns:
            Dictionary mapping post_id to embedding tensor (only valid posts up to max_snapshot).
        """
        # If we've already loaded this snapshot or later, return cached embeddings
        if max_snapshot <= self._loaded_up_to_snapshot:
            return self._post_embeddings

        # Initialize embeddings dict if first load
        if self._post_embeddings is None:
            self._post_embeddings = {}

        if not self._embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self._embeddings_file}")

        # Load all embeddings from disk (pickle doesn't support partial loading)
        if self._loaded_up_to_snapshot == -1:
            print(f"Loading embeddings from {self._embeddings_file}...")
        else:
            print(
                f"Loading additional embeddings for snapshots {self._loaded_up_to_snapshot + 1} to {max_snapshot}..."
            )

        with open(self._embeddings_file, "rb") as f:
            all_embeddings = pickle.load(f)

        total_loaded = len(all_embeddings)

        # Filter to keep only NEW posts that:
        # 1. Are in snapshots we haven't loaded yet (> _loaded_up_to_snapshot)
        # 2. Are in the range we want (<= max_snapshot)
        # 3. Have valid snapshot and user mappings
        # 4. The user appears in our graph
        new_embeddings_count = 0
        for post_id, embedding in all_embeddings.items():
            # Skip if already loaded
            if post_id in self._post_embeddings:
                continue

            # Check validity
            if (
                post_id in self.post_to_snapshot
                and post_id in self.post_to_user
                and self.post_to_user[post_id] in self.user_id_to_node_idx
            ):
                post_snapshot = self.post_to_snapshot[post_id]

                # Only load if in the new snapshot range
                if self._loaded_up_to_snapshot < post_snapshot <= max_snapshot:
                    self._post_embeddings[post_id] = embedding
                    new_embeddings_count += 1

        # Free memory from all embeddings
        del all_embeddings

        # Update tracking
        self._loaded_up_to_snapshot = max_snapshot

        total_kept = len(self._post_embeddings)
        if new_embeddings_count > 0:
            print(
                f"  Added {new_embeddings_count} new embeddings (total: {total_kept}, {100 * total_kept / total_loaded:.1f}% of disk file)"
            )
        else:
            print(f"  No new embeddings to load (total: {total_kept})")

        return self._post_embeddings

    def get_temporal_node_features(self, max_snapshot: int) -> torch.Tensor:
        """Compute node features using only posts up to the given snapshot.

        This allows the model to see how user features evolve over time,
        using only historical information available at each snapshot.

        Args:
            max_snapshot: Maximum snapshot index (inclusive) to use for features.

        Returns:
            Tensor of shape [num_nodes, embedding_dim] with temporal features.
        """
        # Check cache first
        if max_snapshot in self._temporal_features_cache:
            return self._temporal_features_cache[max_snapshot]

        # Load embeddings up to this snapshot (incremental loading)
        post_embeddings = self._load_embeddings_for_snapshot(max_snapshot)

        # Initialize feature matrix
        node_features = torch.zeros(self.num_nodes, self.feats_per_node, dtype=torch.float32)

        # Group posts by user, filtering by snapshot
        user_embeddings: Dict[int, list] = {}

        for post_id, embedding in post_embeddings.items():
            # Check if post exists in our mappings and is within snapshot range
            if post_id in self.post_to_snapshot and post_id in self.post_to_user:
                post_snapshot = self.post_to_snapshot[post_id]

                # Only include posts up to max_snapshot
                if post_snapshot <= max_snapshot:
                    user_id = self.post_to_user[post_id]

                    # Only include users that appear in our graph
                    if user_id in self.user_id_to_node_idx:
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

        # Cache the result
        self._temporal_features_cache[max_snapshot] = node_features

        print(
            f"Computed temporal features for snapshot {max_snapshot}: "
            f"{users_with_features}/{self.num_nodes} nodes with features "
            f"({100 * users_with_features / self.num_nodes:.1f}% coverage)"
        )
        # Optionally unload embeddings if not frequently accessed
        # This saves memory but requires reload if another snapshot is requested
        # Uncomment to enable aggressive memory management:
        # if len(self._temporal_features_cache) > 3:
        #     self.unload_embeddings()
        return node_features

    def clear_temporal_cache(self):
        """Clear the temporal features cache to free memory.

        Call this between training phases or when memory is tight.
        """
        num_cached = len(self._temporal_features_cache)
        self._temporal_features_cache.clear()
        print(f"Cleared {num_cached} cached temporal feature snapshots")

    def unload_embeddings(self):
        """Unload post embeddings from memory to free RAM.

        Embeddings will be reloaded on next access if needed.
        """
        if self._post_embeddings is not None:
            num_embeddings = len(self._post_embeddings)
            self._post_embeddings = None
            print(f"Unloaded {num_embeddings} post embeddings from memory")

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

    def load_and_save_edges(self, bitcoin_args):
        """Load edge data from social network files and save as DataFrame.

        Loads edges from .edg files across multiple time periods and creates
        a DataFrame with columns X, Y, Snapshot, Label (always 1).

        Args:
            bitcoin_args: Configuration with folder path.

        Returns:
            DataFrame with columns ['X', 'Y', 'Snapshot', 'Label'].
        """
        folder_path = Path(bitcoin_args.folder)
        raw_folder = folder_path / "raw"
        processed_folder = folder_path / "processed"
        processed_folder.mkdir(exist_ok=True)

        # Define time periods to load
        time_periods = ["2016-2021", "2022", "2023", "2024", "Jan-Jul 25", "Jul 25"]

        all_edges = []

        for period_idx, period in enumerate(time_periods):
            period_folder = raw_folder / period

            if not period_folder.exists():
                print(f"Warning: Folder {period_folder} does not exist, skipping...")
                continue

            # Load social network edges
            edges_file = period_folder / "social_network.edg"
            if edges_file.exists():
                print(f"Loading {edges_file}...")
                edges = self._load_edg_file(edges_file)

                # Create DataFrame for this snapshot
                # Label is always 1 (positive edge)
                snapshot_df = pd.DataFrame(
                    {
                        "X": edges[:, 0].numpy(),
                        "Y": edges[:, 1].numpy(),
                        "Snapshot": period_idx,
                        "Label": 1,
                    }
                )

                all_edges.append(snapshot_df)
                print(f"  Loaded {len(snapshot_df)} edges from snapshot {period_idx}")
            else:
                print(f"Warning: {edges_file} does not exist, skipping...")

        if not all_edges:
            raise ValueError("No edge files found in the data folder")

        # Concatenate all DataFrames
        edges_df = pd.concat(all_edges, ignore_index=True)

        # Save to file
        output_file = processed_folder / "edges_aggregated.csv"
        edges_df.to_csv(output_file, index=False)
        print(f"\nSaved aggregated edges to {output_file}")
        print(f"Total edges: {len(edges_df)}")
        print(f"Snapshots: {edges_df['Snapshot'].nunique()}")

        return edges_df

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
