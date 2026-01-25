"""Data splitting for train/validation/test sets.

Creates temporal and static data splits for training GCN models.
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import utils as u


class splitter:
    """Creates 3 data splits: train, dev, and test."""

    def __init__(self, args, tasker):
        """Initialize data splitter.

        Args:
            args: Configuration namespace with split ratios and parameters.
            tasker: Tasker object with dataset information.
        """
        assert args.train_proportion + args.dev_proportion < 1, "there's no space for test samples"
        # only the training one requires special handling on start, the others are fine with the split IDX.
        # Convert min_time and max_time to int if they're tensors
        min_time_val = (
            int(tasker.data.min_time.item())
            if torch.is_tensor(tasker.data.min_time)
            else int(tasker.data.min_time)
        )
        max_time_val = (
            int(tasker.data.max_time.item())
            if torch.is_tensor(tasker.data.max_time)
            else int(tasker.data.max_time)
        )

        start = min_time_val + args.num_hist_steps  # -1 + args.adj_mat_time_window
        end = args.train_proportion

        end = int(np.floor(max_time_val * end))
        train = data_split(tasker, start, end, test=False)
        train = DataLoader(train, **args.data_loading_params)

        start = end
        end = args.dev_proportion + args.train_proportion
        end = int(np.floor(max_time_val * end))

        # all_edges: num_nodes * num_nodes - num_positive
        dev = data_split(tasker, start, end, test=True, all_edges=True)

        dev = DataLoader(dev, num_workers=args.data_loading_params["num_workers"])

        start = end

        # the +1 is because I assume that max_time exists in the dataset
        end = max_time_val + 1
        test = data_split(tasker, start, end, test=True, all_edges=True)

        test = DataLoader(test, num_workers=args.data_loading_params["num_workers"])

        print("Dataset splits sizes:  train", len(train), "dev", len(dev), "test", len(test))

        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test


class incremental_splitter:
    """Creates snapshot-based splits for incremental training.

    This splitter divides temporal data into discrete snapshots,
    enabling incremental learning where:
    - Train on snapshot i, test on snapshot i+1
    - Fine-tune on snapshot i+1, test on snapshot i+2
    - And so on...
    """

    def __init__(self, args, tasker):
        """Initialize incremental splitter.

        Args:
            args: Configuration namespace with:
                - num_hist_steps: Number of historical steps for GCN input.
                - adj_mat_time_window: Time window for adjacency matrix.
                - data_loading_params: DataLoader parameters.
                - task: Task type (e.g., 'link_pred').
                - num_snapshots: Number of snapshots to create (optional).
            tasker: Tasker object with dataset information.
        """
        self.args = args
        self.tasker = tasker

        # Convert min_time and max_time to int if they're tensors
        min_time_val = (
            int(tasker.data.min_time.item())
            if torch.is_tensor(tasker.data.min_time)
            else int(tasker.data.min_time)
        )
        max_time_val = (
            int(tasker.data.max_time.item())
            if torch.is_tensor(tasker.data.max_time)
            else int(tasker.data.max_time)
        )

        self.min_time = min_time_val
        self.max_time = max_time_val
        self.num_hist_steps = args.num_hist_steps

        # Calculate valid time range (accounting for history)
        self.valid_start = min_time_val + args.num_hist_steps
        self.valid_end = max_time_val + 1

        # Determine number of snapshots
        # Handle None, 'None', and numeric values
        # if hasattr(args, "num_snapshots") and args.num_snapshots is not None:
        #     if isinstance(args.num_snapshots, str) and args.num_snapshots.lower() == "none":
        #         # Default: each time step is a snapshot
        #         self.num_snapshots = self.valid_end - self.valid_start
        #     else:
        #         self.num_snapshots = int(args.num_snapshots)
        # else:
        #     # Default: each time step is a snapshot
        self.num_snapshots = self.valid_end - self.valid_start

        # Create snapshot boundaries
        self.snapshot_boundaries = self._compute_snapshot_boundaries()
        self.snapshots = self._create_snapshots()

        print(f"\nIncremental Splitter initialized:")
        print(f"  Time range: [{self.valid_start}, {self.valid_end})")
        print(f"  Number of snapshots: {len(self.snapshots)}")
        for i, (start, end) in enumerate(self.snapshot_boundaries):
            print(f"    Snapshot {i}: time [{start}, {end}), size={end - start}")

    def _compute_snapshot_boundaries(self) -> List[tuple]:
        """Compute the time boundaries for each snapshot.

        Returns:
            List of (start, end) tuples for each snapshot.
        """
        total_time_steps = self.valid_end - self.valid_start
        boundaries = []

        if self.num_snapshots >= total_time_steps:
            # Each time step is its own snapshot
            for t in range(self.valid_start, self.valid_end):
                boundaries.append((t, t + 1))
        else:
            # Divide time range into num_snapshots equal parts
            steps_per_snapshot = total_time_steps / self.num_snapshots
            for i in range(self.num_snapshots):
                start = self.valid_start + int(i * steps_per_snapshot)
                end = self.valid_start + int((i + 1) * steps_per_snapshot)
                if i == self.num_snapshots - 1:
                    end = self.valid_end  # Ensure last snapshot includes all remaining
                boundaries.append((start, end))

        return boundaries

    def _create_snapshots(self) -> List[DataLoader]:
        """Create DataLoader for each snapshot.

        Creates two versions of each snapshot:
        - Training version (test=False, normal negative sampling)
        - Testing version (test=True, all_edges for full evaluation)

        Returns:
            List of DataLoaders, one per snapshot (training version).
        """
        train_snapshots = []
        test_snapshots = []

        for start, end in self.snapshot_boundaries:
            # Training version - uses normal negative sampling
            train_data = data_split(self.tasker, start, end, test=False)
            train_loader = DataLoader(
                train_data, batch_size=1, num_workers=self.args.data_loading_params["num_workers"]
            )
            train_snapshots.append(train_loader)

            # Testing version - uses all_edges for full evaluation
            test_data = data_split(self.tasker, start, end, test=True, all_edges=True)

            test_loader = DataLoader(
                test_data, batch_size=1, num_workers=self.args.data_loading_params["num_workers"]
            )
            test_snapshots.append(test_loader)

        # Store both versions
        self._test_snapshots = test_snapshots
        return train_snapshots

    # def get_all_snapshots(self) -> List[DataLoader]:
    #     """Get all snapshot DataLoaders.

    #     Returns:
    #         List of DataLoaders for all snapshots.
    #     """
    #     return self.snapshots

    # def get_snapshot(self, idx: int) -> DataLoader:
    #     """Get a specific snapshot by index.

    #     Args:
    #         idx: Snapshot index.

    #     Returns:
    #         DataLoader for the requested snapshot.

    #     Raises:
    #         IndexError: If idx is out of range.
    #     """
    #     if idx < 0 or idx >= len(self.snapshots):
    #         raise IndexError(f"Snapshot index {idx} out of range [0, {len(self.snapshots)})")
    #     return self.snapshots[idx]

    def get_train_test_pairs(self) -> List[tuple]:
        """Get pairs of (train_snapshot, test_snapshot) for incremental training.

        Returns:
            List of (train_loader, test_loader) tuples.
            - train_loader uses normal negative sampling (test=False)
            - test_loader uses all_edges for comprehensive evaluation
        """
        pairs = []
        for i in range(len(self.snapshots) - 1):
            # Train on snapshot i (training version), test on snapshot i+1 (test version)
            pairs.append((self.snapshots[i], self._test_snapshots[i + 1]))
        return pairs

    def __len__(self) -> int:
        """Get the number of snapshots.

        Returns:
            Number of snapshots.
        """
        return len(self.snapshots)


class data_split(Dataset):
    """Dataset split for temporal data."""

    def __init__(self, tasker, start, end, test, **kwargs):
        """Initialize temporal data split.

        Args:
            tasker: Tasker object.
            start: Starting time index.
            end: Ending time index.
            test: Whether this is a test split.
            **kwargs: Additional arguments for tasker.
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        """Get the number of samples in this split.

        Returns:
            int: Number of samples.
        """
        return self.end - self.start

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Sample dict with input features and labels.
        """
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test=self.test, **self.kwargs)
        return t


# class static_data_split(Dataset):
#     """Dataset split for static graphs."""

#     def __init__(self, tasker, indexes, test):
#         """Initialize static data split.

#         Args:
#             tasker: Tasker object.
#             indexes: Node indices for this split.
#             test: Whether this is a test split.
#         """
#         self.tasker = tasker
#         self.indexes = indexes
#         self.test = test
#         self.adj_matrix = tasker.adj_matrix

#     def __len__(self):
#         """Get the number of samples in this split.

#         Returns:
#             int: Number of samples.
#         """
#         return len(self.indexes)

#     def __getitem__(self, idx):
#         """Get a sample from the dataset.

#         Args:
#             idx: Index of the sample.

#         Returns:
#             Sample dict with input features and labels.
#         """
#         idx = self.indexes[idx]
#         return self.tasker.get_sample(idx, test=self.test)
