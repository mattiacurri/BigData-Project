"""Utilities for EvolveGCN: tensor helpers, argument parsing and seeding."""

import argparse
import math
import random
import time

import numpy as np
import torch
import yaml


def pad_with_last_col(matrix, cols):
    """Pad `matrix` by repeating its last column until it has `cols` columns."""
    out = [matrix]
    pad = [matrix[:, [-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out, dim=1)


def pad_with_last_val(vect, k):
    """Pad `vect` by repeating its last value until length `k`.

    Returns a tensor on the same device as `vect`.
    """
    device = "cuda" if vect.is_cuda else "cpu"
    if vect.size(0) == 0:
        # If vector is empty, pad with zeros
        return torch.zeros(k, dtype=torch.long, device=device)
    pad = torch.ones(k - vect.size(0), dtype=torch.long, device=device) * vect[-1]
    return torch.cat([vect, pad])


def sparse_prepare_tensor(tensor, torch_size, ignore_batch_dim=True):
    """Prepare a sparse tensor dict for PyTorch (optionally remove batch dim)."""
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    return make_sparse_tensor(tensor, tensor_type="float", torch_size=torch_size)


def sp_ignore_batch_dim(tensor_dict):
    """Remove leading batch dimension from sparse-tensor dict fields `idx` and `vals`."""
    tensor_dict["idx"] = tensor_dict["idx"][0]
    tensor_dict["vals"] = tensor_dict["vals"][0]
    return tensor_dict


def aggregate_by_time(time_vector, time_win_aggr):
    """Bucket `time_vector` into windows of size `time_win_aggr` (zero-based)."""
    return (time_vector - time_vector.min()) // time_win_aggr


def sort_by_time(data, time_col):
    """Return `data` sorted by column `time_col`."""
    _, sort = torch.sort(data[:, time_col])
    return data[sort]


def print_sp_tensor(sp_tensor, size):
    """Pretty-print a sparse tensor dict as a dense matrix of shape (size, size)."""
    print(
        torch.sparse.FloatTensor(
            sp_tensor["idx"].t(), sp_tensor["vals"], torch.Size([size, size])
        ).to_dense()
    )


def reset_param(t):
    """Initialize tensor `t` in-place with uniform values using heuristic scale."""
    stdv = 2.0 / math.sqrt(t.size(0))
    t.data.uniform_(-stdv, stdv)


def make_sparse_tensor(adj, tensor_type, torch_size):
    """Create a PyTorch sparse COO tensor from adjacency dict `adj`.

    `tensor_type` must be either "float" or "long".
    """
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size * 2)

    if tensor_type == "float":
        return torch.sparse_coo_tensor(adj["idx"].t(), adj["vals"].type(torch.float), tensor_size)
    if tensor_type == "long":
        return torch.sparse_coo_tensor(adj["idx"].t(), adj["vals"].type(torch.long), tensor_size)
    raise NotImplementedError("only make floats or long sparse tensors")


class Namespace(object):
    """Help reference a mapping as attributes (dict.key instead of dict['key'])."""

    def __init__(self, adict):
        """Initialize Namespace from mapping `adict`."""
        self.__dict__.update(adict)


def set_seeds(rank):
    """Set deterministic seeds for numpy, random and torch based on `rank`."""
    seed = int(time.time()) + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_param_value(param, param_min, param_max, type="int"):
    """Return either `param` or a random value sampled between min/max according to `type`."""
    if str(param) is None or str(param).lower() == "none":
        if type == "int":
            return random.randrange(param_min, param_max + 1)
        if type == "logscale":
            interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        return random.uniform(param_min, param_max)
    return param


def load_data(file):
    """Load CSV-like `file` into a torch tensor (skips header line)."""
    with open(file) as file:
        file = file.read().splitlines()
    return torch.tensor([[float(r) for r in row.split(",")] for row in file[1:]])


def load_data_from_tar(
    file,
    tar_archive,
    replace_unknow=False,
    starting_line=1,
    sep=",",
    type_fn=float,
    tensor_const=torch.DoubleTensor,
):
    """Extract a text file from `tar_archive`, parse rows and return tensor.

    Parameters mirror `load_data` but operate on a file inside a tar.
    """
    f = tar_archive.extractfile(file)
    lines = f.read()  #
    lines = lines.decode("utf-8")
    if replace_unknow:
        lines = lines.replace("unknow", "-1")
        lines = lines.replace("-1n", "-1")

    lines = lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    return tensor_const(data)


def create_parser():
    """Create an argparse.ArgumentParser with the project's standard arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_file",
        default="experiments/parameters_example.yaml",
        type=argparse.FileType(mode="r"),
        help="optional, yaml file containing parameters to be used, overrides command line parameters",
    )
    return parser


def parse_args(parser):
    """Parse CLI args, optionally load YAML config and sample randomized hyperparameters."""
    args = parser.parse_args()
    if args.config_file:
        config_file_path = args.config_file.name
        config_file_name = (
            args.config_file.name.split("/")[-1]
            .split("\\")[-1]
            .replace(".yaml", "")
            .replace(".yml", "")
        )
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        arg_dict["config_file_name"] = config_file_name
        arg_dict["config_file_path"] = config_file_path
        for key, value in data.items():
            arg_dict[key] = value
    else:
        args.config_file_name = "default"
        args.config_file_path = None

    args.learning_rate = random_param_value(
        args.learning_rate,
        args.learning_rate_min,
        args.learning_rate_max,
        type="logscale",
    )
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')
    args.num_hist_steps = random_param_value(
        args.num_hist_steps,
        args.num_hist_steps_min,
        args.num_hist_steps_max,
        type="int",
    )
    args.gcn_parameters["feats_per_node"] = random_param_value(
        args.gcn_parameters["feats_per_node"],
        args.gcn_parameters["feats_per_node_min"],
        args.gcn_parameters["feats_per_node_max"],
        type="int",
    )
    args.gcn_parameters["layer_1_feats"] = random_param_value(
        args.gcn_parameters["layer_1_feats"],
        args.gcn_parameters["layer_1_feats_min"],
        args.gcn_parameters["layer_1_feats_max"],
        type="int",
    )
    args.gcn_parameters["layer_2_feats"] = random_param_value(
        args.gcn_parameters["layer_2_feats"],
        args.gcn_parameters["layer_1_feats_min"],
        args.gcn_parameters["layer_1_feats_max"],
        type="int",
    )
    args.gcn_parameters["cls_feats"] = random_param_value(
        args.gcn_parameters["cls_feats"],
        args.gcn_parameters["cls_feats_min"],
        args.gcn_parameters["cls_feats_max"],
        type="int",
    )

    if not hasattr(args, "use_wandb"):
        args.use_wandb = False
    if not hasattr(args, "wandb_project"):
        args.wandb_project = "evolvegcn"
    if not hasattr(args, "wandb_entity"):
        args.wandb_entity = None
    if not hasattr(args, "wandb_run_name"):
        args.wandb_run_name = None
    if not hasattr(args, "wandb_log_minibatch"):
        args.wandb_log_minibatch = False

    if not hasattr(args, "split_mode"):
        args.split_mode = "proportion"
    if not hasattr(args, "loocv_valid_snapshot"):
        args.loocv_valid_snapshot = None
    if not hasattr(args, "loocv_test_snapshot"):
        args.loocv_test_snapshot = None

    return args
