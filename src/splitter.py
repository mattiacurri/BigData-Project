"""Data splitting for train/validation/test sets.

Creates temporal and static data splits for training GCN models.
"""

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
        if tasker.is_static:  #### For static datsets
            assert args.train_proportion + args.dev_proportion < 1, (
                "there's no space for test samples"
            )
            # only the training one requires special handling on start, the others are fine with the split IDX.

            random_perm = False
            indexes = tasker.data.nodes_with_label

            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print("tasker.data.nodes", indexes.size())
                perm_idx, _ = indexes.sort()

            self.train_idx = perm_idx[: int(args.train_proportion * perm_idx.size(0))]
            self.dev_idx = perm_idx[
                int(args.train_proportion * perm_idx.size(0)) : int(
                    (args.train_proportion + args.dev_proportion) * perm_idx.size(0)
                )
            ]
            self.test_idx = perm_idx[
                int((args.train_proportion + args.dev_proportion) * perm_idx.size(0)) :
            ]

            train = static_data_split(tasker, self.train_idx, test=False)
            train = DataLoader(train, shuffle=True, **args.data_loading_params)

            dev = static_data_split(tasker, self.dev_idx, test=True)
            dev = DataLoader(dev, shuffle=False, **args.data_loading_params)

            test = static_data_split(tasker, self.test_idx, test=True)
            test = DataLoader(test, shuffle=False, **args.data_loading_params)

            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test

        else:  #### For datsets with time
            assert args.train_proportion + args.dev_proportion < 1, (
                "there's no space for test samples"
            )
            # only the training one requires special handling on start, the others are fine with the split IDX.
            start = tasker.data.min_time + args.num_hist_steps  # -1 + args.adj_mat_time_window
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

            # the +1 is because I assume that max_time exists in the dataset
            end = int(tasker.max_time) + 1
            if args.task == "link_pred":
                test = data_split(tasker, start, end, test=True, all_edges=True)
            else:
                test = data_split(tasker, start, end, test=True)

            test = DataLoader(test, num_workers=args.data_loading_params["num_workers"])

            print("Dataset splits sizes:  train", len(train), "dev", len(dev), "test", len(test))

            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test


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


class static_data_split(Dataset):
    """Dataset split for static graphs."""

    def __init__(self, tasker, indexes, test):
        """Initialize static data split.

        Args:
            tasker: Tasker object.
            indexes: Node indices for this split.
            test: Whether this is a test split.
        """
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        """Get the number of samples in this split.

        Returns:
            int: Number of samples.
        """
        return len(self.indexes)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Sample dict with input features and labels.
        """
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx, test=self.test)
