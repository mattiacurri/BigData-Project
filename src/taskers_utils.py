"""Utility functions for taskers.

Provides functions for graph manipulation, feature generation, and edge sampling.
"""

import time

import numpy as np
import torch

import utils as u

ECOLS = u.Namespace({"source": 0, "target": 1, "time": 2, "label": 3})  # --> added for edge_cls


def get_sp_adj(edges, time, weighted, time_window):
    """Get sparse adjacency matrix for a specific time window.

    Args:
        edges: Edge list with timestamps.
        time: Target time.
        weighted: Whether to include edge weights.
        time_window: Size of time window.

    Returns:
        Dict with adjacency matrix in sparse format.
    """
    idx = edges["idx"]
    subset = idx[:, ECOLS.time] <= time
    subset = subset * (idx[:, ECOLS.time] > (time - time_window))
    idx = edges["idx"][subset][:, [ECOLS.source, ECOLS.target]]
    vals = edges["vals"][subset]
    out = torch.sparse_coo_tensor(idx.t(), vals).coalesce()

    idx = out._indices().t()
    if weighted:
        vals = out._values()
    else:
        vals = torch.ones(idx.size(0), dtype=torch.long)

    return {"idx": idx, "vals": vals}


def get_node_mask(cur_adj, num_nodes):
    """Get node mask for masking non-existent nodes.

    Args:
        cur_adj: Current adjacency matrix.
        num_nodes: Total number of nodes.

    Returns:
        Mask tensor with -inf for masked nodes, 0 otherwise.
    """
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj["idx"].unique()

    mask[non_zero] = 0

    return mask


def normalize_adj(adj, num_nodes):
    """Normalize adjacency matrix using symmetric normalization.

    Applies: A_norm = D^{-1/2} (A + I) D^{-1/2}

    Args:
        adj: Adjacency dict with 'idx' and 'vals'.
        num_nodes: Number of nodes.

    Returns:
        Normalized adjacency dict.
    """
    idx = adj["idx"]
    vals = adj["vals"]

    sp_tensor = torch.sparse_coo_tensor(
        idx.t(), vals.type(torch.float), size=(num_nodes, num_nodes)
    )

    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)

    return {"idx": idx.t(), "vals": vals}


def make_sparse_eye(size):
    """Create sparse identity matrix.

    Args:
        size: Dimension of the identity matrix.

    Returns:
        Sparse identity tensor.
    """
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx, eye_idx], dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse_coo_tensor(eye_idx, vals, size=(size, size))
    return eye


def get_all_non_existing_edges(adj, tot_nodes):
    """Get all non-existing edges in the graph.

    Args:
        adj: Adjacency dict with existing edges.
        tot_nodes: Total number of nodes.

    Returns:
        Dict with all non-existing edge pairs.
    """
    true_ids = adj["idx"].t().numpy()
    true_ids_set = set(get_edges_ids(true_ids, tot_nodes))

    # total_possible = tot_nodes * tot_nodes - num_positive

    all_edges_idx = np.arange(tot_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx, all_edges_idx)).reshape(2, -1)

    all_edges_ids = get_edges_ids(all_edges_idx, tot_nodes)

    # only edges that are not in the true_ids should keep here
    mask = np.logical_not(np.isin(all_edges_ids, true_ids_set))

    non_existing_edges_idx = all_edges_idx[:, mask]
    edges = torch.tensor(non_existing_edges_idx).t()
    vals = torch.zeros(edges.size(0), dtype=torch.long)
    return {"idx": edges, "vals": vals}


def _sample_non_existing_edges_efficient(true_ids_set, tot_nodes, num_samples):
    """Efficiently sample non-existing edges without generating all possibilities.

    Args:
        true_ids_set: Set of existing edge IDs.
        tot_nodes: Total number of nodes.
        num_samples: Number of negative edges to sample.

    Returns:
        Dict with sampled non-existing edges.
    """
    sampled = set()
    attempts = 0
    max_attempts = num_samples * 10  # Avoid infinite loop

    while len(sampled) < num_samples and attempts < max_attempts:
        # Sample batch of random edges
        batch_size = min(num_samples * 2, 1000000)
        src = np.random.randint(0, tot_nodes, batch_size)
        dst = np.random.randint(0, tot_nodes, batch_size)

        for s, d in zip(src, dst):
            edge_id = s * tot_nodes + d
            # Skip self-loops and existing edges
            if s != d and edge_id not in true_ids_set and edge_id not in sampled:
                sampled.add(edge_id)
                if len(sampled) >= num_samples:
                    break
        attempts += batch_size

    # Convert back to edge indices
    sampled_ids = np.array(list(sampled))
    src_nodes = sampled_ids // tot_nodes
    dst_nodes = sampled_ids % tot_nodes

    edges = torch.tensor(np.stack([src_nodes, dst_nodes], axis=1), dtype=torch.long)
    vals = torch.zeros(edges.size(0), dtype=torch.long)
    return {"idx": edges, "vals": vals}


def get_non_existing_edges(adj, number, tot_nodes, smart_sampling, existing_nodes=None):
    """Sample non-existing edges for negative sampling.

    Args:
        adj: Adjacency dict with existing edges.
        number: Number of non-existing edges to sample.
        tot_nodes: Total number of nodes.
        smart_sampling: Whether to sample from existing node neighborhoods.
        existing_nodes: Nodes with existing edges (for smart sampling).

    Returns:
        Dict with sampled non-existing edges.
    """
    t0 = time.time()
    idx = adj["idx"].t().numpy()
    true_ids = get_edges_ids(idx, tot_nodes)

    true_ids = set(true_ids)

    # the maximum of edges would be all edges that don't exist between nodes that have edges
    num_edges = min(number, idx.shape[1] * (idx.shape[1] - 1) - len(true_ids))

    if smart_sampling:
        # existing_nodes = existing_nodes.numpy()
        def sample_edges(num_edges):
            from_id = np.random.choice(idx[0], size=num_edges, replace=True)
            to_id = np.random.choice(existing_nodes, size=num_edges, replace=True)

            if num_edges > 1:
                edges = np.stack([from_id, to_id])
            else:
                edges = np.concatenate([from_id, to_id])
            return edges
    else:

        def sample_edges(num_edges):
            if num_edges > 1:
                edges = np.random.randint(0, tot_nodes, (2, num_edges))
            else:
                edges = np.random.randint(0, tot_nodes, (2,))
            return edges

    edges = sample_edges(num_edges * 4)

    edge_ids = edges[0] * tot_nodes + edges[1]

    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(num_edges * 4):
        eid = edge_ids[i]
        # ignore if any of these conditions happen
        if eid in out_ids or edges[0, i] == edges[1, i] or eid in true_ids:
            continue

        # add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1

        # if we have sampled enough edges break
        if num_sampled >= num_edges:
            break

    edges = edges[:, sampled_indices]
    edges = torch.tensor(edges).t()
    vals = torch.zeros(edges.size(0), dtype=torch.long)
    return {"idx": edges, "vals": vals}


def get_edges_ids(sp_idx, tot_nodes):
    """Convert edge indices to linear indices.

    Args:
        sp_idx: Edge indices array.
        tot_nodes: Total number of nodes.

    Returns:
        Array of linear edge IDs.
    """
    return sp_idx[0] * tot_nodes + sp_idx[1]
