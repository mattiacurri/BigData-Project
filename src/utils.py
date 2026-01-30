"""Various utility functions.

Taken from: https://github.com/IBM/EvolveGCN/blob/master/utils.py
"""

import torch
import yaml


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
        # remove batch dimension
        tensor = {"idx": tensor["idx"][0], "vals": tensor["vals"][0]}

    tensor_size = torch.Size(torch_size) if len(torch_size) == 2 else torch.Size(torch_size * 2)

    return torch.sparse_coo_tensor(
        tensor["idx"].t(), tensor["vals"].type(torch.float), tensor_size
    )


class Namespace(object):
    """Helper class for dictionary-like attribute access."""

    def __init__(self, adict):
        """Initialize namespace from dictionary.

        Args:
            adict: Dictionary to convert to namespace.
        """
        self.__dict__.update(adict)


def parse_args(parser):
    """Parse command line arguments and YAML config file.

    Args:
        parser: ArgumentParser object.

    Returns:
        Namespace with parsed arguments.
    """
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    return args
