"""Various utility functions.

Taken from: https://github.com/IBM/EvolveGCN/blob/master/utils.py
"""

import argparse
import math
import random
import time

import numpy as np
import torch
import yaml


def pad_with_last_col(matrix, cols):
    """Pad matrix columns with the last column repeated.

    Args:
        matrix: Input matrix to pad.
        cols: Target number of columns.

    Returns:
        Padded matrix with shape (rows, cols).
    """
    out = [matrix]
    pad = [matrix[:, [-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out, dim=1)


def pad_with_last_val(vect, k):
    """Pad vector to length k with the last value repeated.

    Args:
        vect: Input vector.
        k: Target length.

    Returns:
        Padded vector of length k.
    """
    device = "cuda" if vect.is_cuda else "cpu"
    pad = torch.ones(k - vect.size(0), dtype=torch.long, device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


def sparse_prepare_tensor(tensor, torch_size, ignore_batch_dim=True):
    """Prepare a sparse tensor for model input.

    Args:
        tensor: Input tensor dict with 'idx' and 'vals'.
        torch_size: Target size of the sparse tensor.
        ignore_batch_dim: Whether to ignore batch dimension.

    Returns:
        Sparse tensor of specified size.
    """
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor, tensor_type="float", torch_size=torch_size)
    return tensor


def sp_ignore_batch_dim(tensor_dict):
    """Remove batch dimension from sparse tensor.

    Args:
        tensor_dict: Tensor dict with 'idx' and 'vals'.

    Returns:
        Tensor dict with batch dimension removed.
    """
    tensor_dict["idx"] = tensor_dict["idx"][0]
    tensor_dict["vals"] = tensor_dict["vals"][0]
    return tensor_dict


def aggregate_by_time(time_vector, time_win_aggr):
    """Aggregate time steps by specified window size.

    Args:
        time_vector: Vector of timestamps.
        time_win_aggr: Time window for aggregation.

    Returns:
        Aggregated time vector.
    """
    time_vector = time_vector - time_vector.min()
    time_vector = time_vector // time_win_aggr
    return time_vector


def sort_by_time(data, time_col):
    """Sort data by time column.

    Args:
        data: Input data tensor.
        time_col: Index of time column.

    Returns:
        Data sorted by time column.
    """
    _, sort = torch.sort(data[:, time_col])
    data = data[sort]
    return data


def print_sp_tensor(sp_tensor, size):
    """Print a sparse tensor in dense format.

    Args:
        sp_tensor: Sparse tensor dict.
        size: Dimension size of the sparse tensor.
    """
    print(
        torch.sparse.FloatTensor(
            sp_tensor["idx"].t(), sp_tensor["vals"], torch.Size([size, size])
        ).to_dense()
    )


def reset_param(t):
    """Reset parameter values uniformly.

    Args:
        t: Parameter tensor to reset.
    """
    stdv = 2.0 / math.sqrt(t.size(0))
    t.data.uniform_(-stdv, stdv)


def make_sparse_tensor(adj, tensor_type, torch_size):
    """Create a sparse tensor from adjacency dict.

    Args:
        adj: Adjacency dict with 'idx' and 'vals'.
        tensor_type: Type ('float' or 'long').
        torch_size: Target size.

    Returns:
        Sparse tensor of specified type.
    """
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size * 2)

    if tensor_type == "float":
        test = torch.sparse.FloatTensor(adj["idx"].t(), adj["vals"].type(torch.float), tensor_size)
        return torch.sparse.FloatTensor(adj["idx"].t(), adj["vals"].type(torch.float), tensor_size)
    elif tensor_type == "long":
        return torch.sparse.LongTensor(adj["idx"].t(), adj["vals"].type(torch.long), tensor_size)
    else:
        raise NotImplementedError("only make floats or long sparse tensors")


def sp_to_dict(sp_tensor):
    """Convert sparse tensor to dictionary format.

    Args:
        sp_tensor: PyTorch sparse tensor.

    Returns:
        Dict with 'idx' and 'vals' keys.
    """
    return {"idx": sp_tensor._indices().t(), "vals": sp_tensor._values()}


class Namespace(object):
    """Helper class for dictionary-like attribute access."""

    def __init__(self, adict):
        """Initialize namespace from dictionary.

        Args:
            adict: Dictionary to convert to namespace.
        """
        self.__dict__.update(adict)


def random_param_value(param, param_min, param_max, type="int"):
    """Sample a random parameter value from a range.

    Args:
        param: Parameter value (if not None, returned as-is).
        param_min: Minimum parameter value.
        param_max: Maximum parameter value.
        type: Type of sampling ('int', 'logscale', or 'float').

    Returns:
        Sampled parameter value.
    """
    if str(param) is None or str(param).lower() == "none":
        if type == "int":
            return random.randrange(param_min, param_max + 1)
        elif type == "logscale":
            interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param


def load_data(file):
    """Load data from CSV file.

    Args:
        file: Path to CSV file.

    Returns:
        Tensor of data.
    """
    with open(file) as file:
        file = file.read().splitlines()
    data = torch.tensor([[float(r) for r in row.split(",")] for row in file[1:]])
    return data


def load_data_from_tar(
    file,
    tar_archive,
    replace_unknow=False,
    starting_line=1,
    sep=",",
    type_fn=float,
    tensor_const=torch.DoubleTensor,
):
    """Load data from a file within a tar archive.

    Args:
        file: File path within tar archive.
        tar_archive: Tar archive object.
        replace_unknow: Whether to replace 'unknow' strings with -1.
        starting_line: Line to start reading from.
        sep: Field separator.
        type_fn: Type conversion function.
        tensor_const: Tensor constructor.

    Returns:
        Tensor of data.
    """
    f = tar_archive.extractfile(file)
    lines = f.read()  #
    lines = lines.decode("utf-8")
    if replace_unknow:
        lines = lines.replace("unknow", "-1")
        lines = lines.replace("-1n", "-1")

    lines = lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    return data


def create_parser():
    """Create argument parser for experiment configuration.

    Returns:
        ArgumentParser object.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_file",
        default="experiments/parameters_example.yaml",
        type=argparse.FileType(mode="r"),
        help="optional, yaml file containing parameters to be used, overrides command line parameters",
    )
    return parser


def parse_args(parser):
    """Parse command line arguments and YAML config file.

    Args:
        parser: ArgumentParser object.

    Returns:
        Namespace with parsed arguments.
    """
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    args.learning_rate = random_param_value(
        args.learning_rate, args.learning_rate_min, args.learning_rate_max, type="logscale"
    )
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')
    args.num_hist_steps = random_param_value(
        args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type="int"
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
    if (
        args.gcn_parameters["layer_2_feats_same_as_l1"]
        or args.gcn_parameters["layer_2_feats_same_as_l1"].lower() == "true"
    ):
        args.gcn_parameters["layer_2_feats"] = args.gcn_parameters["layer_1_feats"]
    else:
        args.gcn_parameters["layer_2_feats"] = random_param_value(
            args.gcn_parameters["layer_2_feats"],
            args.gcn_parameters["layer_1_feats_min"],
            args.gcn_parameters["layer_1_feats_max"],
            type="int",
        )
    args.gcn_parameters["lstm_l1_feats"] = random_param_value(
        args.gcn_parameters["lstm_l1_feats"],
        args.gcn_parameters["lstm_l1_feats_min"],
        args.gcn_parameters["lstm_l1_feats_max"],
        type="int",
    )
    if (
        args.gcn_parameters["lstm_l2_feats_same_as_l1"]
        or args.gcn_parameters["lstm_l2_feats_same_as_l1"].lower() == "true"
    ):
        args.gcn_parameters["lstm_l2_feats"] = args.gcn_parameters["lstm_l1_feats"]
    else:
        args.gcn_parameters["lstm_l2_feats"] = random_param_value(
            args.gcn_parameters["lstm_l2_feats"],
            args.gcn_parameters["lstm_l1_feats_min"],
            args.gcn_parameters["lstm_l1_feats_max"],
            type="int",
        )
    args.gcn_parameters["cls_feats"] = random_param_value(
        args.gcn_parameters["cls_feats"],
        args.gcn_parameters["cls_feats_min"],
        args.gcn_parameters["cls_feats_max"],
        type="int",
    )

    return args
