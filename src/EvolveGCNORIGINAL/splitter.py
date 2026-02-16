"""Data split utilities returning PyTorch Dataset/DataLoader objects."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class data_split(Dataset):
    """Dataset that yields samples for a time range [start, end)."""

    def __init__(self, tasker, start, end, test, **kwargs):
        """Start and end are indices indicating what items belong to this split."""
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        """Return number of items in this split."""
        return self.end - self.start

    def __getitem__(self, idx):
        """Return sample at position `idx` (offset by `start`)."""
        idx = self.start + idx
        return self.tasker.get_sample(idx, test=self.test, **self.kwargs)


class snapshot_list_split(Dataset):
    """Dataset built from an explicit list of snapshots."""

    def __init__(self, tasker, snapshots, test):
        """`snapshots` is a sorted list of snapshot indices to include."""
        self.tasker = tasker
        self.snapshots = sorted(snapshots)
        self.test = test

    def __len__(self):
        """Return number of snapshots in this split."""
        return len(self.snapshots)

    def __getitem__(self, idx):
        """Return sample for snapshot at `idx`."""
        snapshot = self.snapshots[idx]
        return self.tasker.get_sample(snapshot, test=self.test)


class static_data_split(Dataset):
    """Dataset wrapper for static node-index based sampling."""

    def __init__(self, tasker, indexes, test):
        """Start and end are indices indicating what items belong to this split."""
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        """Return number of items in `indexes`."""
        return len(self.indexes)

    def __getitem__(self, idx):
        """Return sample corresponding to `indexes[idx]`."""
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx, test=self.test)


class splitter:
    """Create three dataset splits: train, dev and test.

    Supports two split modes:
    - "proportion": split by train_proportion and dev_proportion (default).
    - "loocv": leave-one-out cross-validation on snapshots.
    """

    def __init__(self, args, tasker):
        """Initialize `splitter` with `args` and `tasker`."""
        self.tasker = tasker
        self.args = args

        split_mode = getattr(args, "split_mode", "proportion")

        if split_mode == "loocv":
            self._init_loocv_split(args, tasker)
        else:
            self._init_proportion_split(args, tasker)

    def _init_proportion_split(self, args, tasker):
        assert args.train_proportion + args.dev_proportion <= 1, (
            "there's no space for test samples"
        )
        start = tasker.data.min_time + args.num_hist_steps
        end = args.train_proportion

        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(tasker, start, end, test=False)
        train = DataLoader(train, **args.data_loading_params)

        start = end
        end = args.dev_proportion + args.train_proportion
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        if args.task == "link_pred":
            dev = data_split(tasker, start, end, test=True, all_edges=True)
        else:
            dev = data_split(tasker, start, end, test=True)

        dev = DataLoader(dev, num_workers=args.data_loading_params["num_workers"])

        start = end

        end = int(tasker.data.max_time) + 1
        if args.task == "link_pred":
            test = data_split(tasker, start, end, test=True, all_edges=True)
        else:
            test = data_split(tasker, start, end, test=True)

        test = DataLoader(test, num_workers=args.data_loading_params["num_workers"])

        print(
            "Dataset splits sizes:  train",
            len(train),
            "dev",
            len(dev),
            "test",
            len(test),
        )

        self.train = train
        self.dev = dev
        self.test = test

        end_train = int(np.floor(tasker.data.max_time.type(torch.float) * args.train_proportion))
        start_dev = end_train
        end_dev = int(
            np.floor(
                tasker.data.max_time.type(torch.float)
                * (args.dev_proportion + args.train_proportion)
            )
        )
        start_test = end_dev
        end_test = int(tasker.data.max_time) + 1

        self.train_idx = self.get_unique_nodes(tasker.data.min_time, end_train)
        self.dev_idx = self.get_unique_nodes(start_dev, end_dev)
        self.test_idx = self.get_unique_nodes(start_test, end_test)

        self._print_node_distribution(tasker, end_train, start_dev, end_dev, start_test, end_test)

    def _init_loocv_split(self, args, tasker):
        min_valid_time = tasker.data.min_time + args.num_hist_steps
        max_time = int(tasker.data.max_time)

        valid_snapshot = args.loocv_valid_snapshot
        test_snapshot = args.loocv_test_snapshot

        assert valid_snapshot is not None, "loocv_valid_snapshot must be specified"
        assert test_snapshot is not None, "loocv_test_snapshot must be specified"
        assert valid_snapshot >= min_valid_time, (
            f"loocv_valid_snapshot ({valid_snapshot}) must be >= min_time + num_hist_steps ({min_valid_time})"
        )
        assert test_snapshot >= min_valid_time, (
            f"loocv_test_snapshot ({test_snapshot}) must be >= min_time + num_hist_steps ({min_valid_time})"
        )
        assert valid_snapshot <= max_time, (
            f"loocv_valid_snapshot ({valid_snapshot}) must be <= max_time ({max_time})"
        )
        assert test_snapshot <= max_time, (
            f"loocv_test_snapshot ({test_snapshot}) must be <= max_time ({max_time})"
        )

        all_snapshots = list(range(min_valid_time, max_time + 1))
        excluded = set([valid_snapshot, test_snapshot])
        train_snapshots = [s for s in all_snapshots if s not in excluded]

        print("LOOCV split mode:")
        print(f"  All available snapshots: {all_snapshots}")
        print(f"  Train snapshots: {train_snapshots}")
        print(f"  Valid snapshot: {valid_snapshot}")
        print(f"  Test snapshot: {test_snapshot}")

        train = snapshot_list_split(tasker, train_snapshots, test=False)
        train = DataLoader(train, **args.data_loading_params)

        dev = snapshot_list_split(tasker, [valid_snapshot], test=True)
        dev = DataLoader(dev, num_workers=args.data_loading_params["num_workers"])

        test = snapshot_list_split(tasker, [test_snapshot], test=True)
        test = DataLoader(test, num_workers=args.data_loading_params["num_workers"])

        print(
            "Dataset splits sizes:  train",
            len(train),
            "dev",
            len(dev),
            "test",
            len(test),
        )

        self.train = train
        self.dev = dev
        self.test = test

        self.train_idx = self.get_unique_nodes_for_snapshots(train_snapshots)
        self.dev_idx = self.get_unique_nodes_for_snapshots([valid_snapshot])
        self.test_idx = self.get_unique_nodes_for_snapshots([test_snapshot])

        print(f"\n{'=' * 60}")
        print("NODE DISTRIBUTION ACROSS SPLITS (LOOCV):")
        print(f"  Train nodes: {len(self.train_idx)} (snapshots {train_snapshots})")
        print(f"  Dev nodes: {len(self.dev_idx)} (snapshot {valid_snapshot})")
        print(f"  Test nodes: {len(self.test_idx)} (snapshot {test_snapshot})")

        train_set = set(self.train_idx)
        dev_set = set(self.dev_idx)
        test_set = set(self.test_idx)

        new_in_dev = len(dev_set - train_set)
        new_in_test = len(test_set - train_set - dev_set)

        print(f"\n  New nodes in dev (not in train): {new_in_dev}")
        print(f"  New nodes in test (not in train/dev): {new_in_test}")
        print(f"{'=' * 60}\n")

    def _print_node_distribution(
        self, tasker, end_train, start_dev, end_dev, start_test, end_test
    ):
        print(f"\n{'=' * 60}")
        print("NODE DISTRIBUTION ACROSS SPLITS:")
        print(f"  Train nodes: {len(self.train_idx)} (time {tasker.data.min_time}-{end_train})")
        print(f"  Dev nodes: {len(self.dev_idx)} (time {start_dev}-{end_dev})")
        print(f"  Test nodes: {len(self.test_idx)} (time {start_test}-{end_test})")

        train_set = set(self.train_idx)
        dev_set = set(self.dev_idx)
        test_set = set(self.test_idx)

        new_in_dev = len(dev_set - train_set)
        new_in_test = len(test_set - train_set - dev_set)

        print(f"\n  New nodes in dev (not in train): {new_in_dev}")
        print(f"  New nodes in test (not in train/dev): {new_in_test}")
        print(f"{'=' * 60}\n")

    def get_unique_nodes(self, start, end):
        """Return unique node ids that appear in edges with time in [start, end)."""
        mask = (self.tasker.data.edges["idx"][:, 2] >= start) & (
            self.tasker.data.edges["idx"][:, 2] < end
        )
        edges_in_range = self.tasker.data.edges["idx"][mask]
        if edges_in_range.numel() == 0:
            return []
        nodes = torch.unique(edges_in_range[:, :2].flatten())
        return nodes.tolist()

    def get_unique_nodes_for_snapshots(self, snapshots):
        """Return unique node ids that appear across `snapshots`."""
        all_nodes = []
        for t in snapshots:
            mask = self.tasker.data.edges["idx"][:, 2] == t
            edges_at_t = self.tasker.data.edges["idx"][mask]
            if edges_at_t.numel() > 0:
                nodes = torch.unique(edges_at_t[:, :2].flatten())
                all_nodes.extend(nodes.tolist())
        return list(set(all_nodes))
