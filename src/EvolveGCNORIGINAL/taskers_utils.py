"""Tasker utility helpers: degree features, adjacency helpers and negative sampling."""

import numpy as np
import torch

import utils as u

ECOLS = u.Namespace({"source": 0, "target": 1, "time": 2, "label": 3})  # --> added for edge_cls

# def get_2_hot_deg_feats(adj,max_deg_out,max_deg_in,num_nodes):
#     #For now it'll just return a 2-hot vector
#     adj['vals'] = torch.ones(adj['idx'].size(0))
#     degs_out, degs_in = get_degree_vects(adj,num_nodes)

#     degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_out.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}

#     # print ('XXX degs_out',degs_out['idx'].size(),degs_out['vals'].size())
#     degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg_out])

#     degs_in = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_in.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}
#     degs_in = u.make_sparse_tensor(degs_in,'long',[num_nodes,max_deg_in])

#     hot_2 = torch.cat([degs_out,degs_in],dim = 1)
#     hot_2 = {'idx': hot_2._indices().t(),
#              'vals': hot_2._values()}

#     return hot_2


def get_1_hot_deg_feats(adj, max_deg, num_nodes):
    """Return 1-hot degree features for nodes from sparse `adj`."""
    # For now it'll just return a 2-hot vector
    new_vals = torch.ones(adj["idx"].size(0))
    new_adj = {"idx": adj["idx"], "vals": new_vals}
    degs_out, _ = get_degree_vects(new_adj, num_nodes)

    degs_out = {
        "idx": torch.cat([torch.arange(num_nodes).view(-1, 1), degs_out.view(-1, 1)], dim=1),
        "vals": torch.ones(num_nodes),
    }

    degs_out = u.make_sparse_tensor(degs_out, "long", [num_nodes, max_deg])

    return {"idx": degs_out._indices().t(), "vals": degs_out._values()}


def get_max_degs(args, dataset, all_window=False, up_to_time=None):
    """Compute maximum in/out degrees over time windows for the dataset."""
    max_deg_out = []
    max_deg_in = []
    max_t = up_to_time if up_to_time is not None else dataset.max_time
    for t in range(dataset.min_time, max_t):
        window = t + 1 if all_window else args.adj_mat_time_window

        cur_adj = get_sp_adj(edges=dataset.edges, time=t, weighted=False, time_window=window)
        cur_out, cur_in = get_degree_vects(cur_adj, dataset.num_nodes)
        max_deg_out.append(cur_out.max())
        max_deg_in.append(cur_in.max())
        # max_deg_out = torch.stack([max_deg_out,cur_out.max()]).max()
        # max_deg_in = torch.stack([max_deg_in,cur_in.max()]).max()
    if max_deg_out:
        max_deg_out = torch.stack(max_deg_out).max()
        max_deg_in = torch.stack(max_deg_in).max()
        max_deg_out = int(max_deg_out) + 1
        max_deg_in = int(max_deg_in) + 1
    else:
        max_deg_out = 1
        max_deg_in = 1

    return max_deg_out, max_deg_in


def get_max_degs_static(num_nodes, adj_matrix):
    """Return maximum out/in degree from a static adjacency matrix."""
    cur_out, cur_in = get_degree_vects(adj_matrix, num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1
    max_deg_in = int(cur_in.max().item()) + 1

    return max_deg_out, max_deg_in


def get_degree_vects(adj, num_nodes):
    """Compute out- and in-degree vectors from sparse adjacency `adj`."""
    adj = u.make_sparse_tensor(adj, "long", [num_nodes])
    degs_out = adj.matmul(torch.ones(num_nodes, 1, dtype=torch.long))
    degs_in = adj.t().matmul(torch.ones(num_nodes, 1, dtype=torch.long))
    return degs_out, degs_in


def get_sp_adj(edges, time, weighted, time_window, cumulative=False):
    """Return sparse adjacency (idx/vals) for `time` using `time_window`.

    If `cumulative` is True the window is [0, time], otherwise (time-time_window, time].
    """
    idx = edges["idx"]
    if cumulative:
        subset = idx[:, ECOLS.time] <= time
    else:
        subset = idx[:, ECOLS.time] <= time
        subset = subset * (idx[:, ECOLS.time] > (time - time_window))
    idx = edges["idx"][subset][:, [ECOLS.source, ECOLS.target]]
    vals = edges["vals"][subset]
    out = torch.sparse_coo_tensor(idx.t(), vals).coalesce()

    idx = out.indices().t()
    vals = out.values() if weighted else torch.ones(idx.size(0), dtype=torch.long)

    return {"idx": idx, "vals": vals}


def get_edge_labels(edges, time):
    """Return labels for edges occurring at `time` as a dict with `idx` and `vals`."""
    idx = edges["idx"]
    subset = idx[:, ECOLS.time] == time
    idx = edges["idx"][subset][:, [ECOLS.source, ECOLS.target]]
    vals = edges["idx"][subset][:, ECOLS.label]

    return {"idx": idx, "vals": vals}


def get_node_mask(cur_adj, num_nodes):
    """Return a mask indicating which nodes have at least one edge in `cur_adj`."""
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj["idx"].unique()

    mask[non_zero] = 0

    return mask


def get_sp_adj_only_new(edges, time, weighted):
    """Return sparse adjacency for `time` considering only newly appeared edges."""
    return get_sp_adj(edges, time, weighted, time_window=1)


def normalize_adj(adj, num_nodes):
    """Normalize adjacency dict (idx/vals).

    Takes an adj matrix as a dict with `idx` and `vals` and normalizes it by:
    - adding an identity matrix,
    - computing the degree vector,
    - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2.
    """
    idx = adj["idx"]
    vals = adj["vals"]

    sp_tensor = torch.sparse_coo_tensor(idx.t(), vals.type(torch.float), (num_nodes, num_nodes))

    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = (sparse_eye + sp_tensor).coalesce()

    idx = sp_tensor.indices()
    vals = sp_tensor.values()

    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)

    return {"idx": idx.t(), "vals": vals}


def make_sparse_eye(size):
    """Return a sparse identity matrix of given `size`."""
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx, eye_idx], dim=1).t()
    vals = torch.ones(size)
    return torch.sparse_coo_tensor(eye_idx, vals, (size, size))


def get_all_non_existing_edges(adj, tot_nodes):
    """Return all possible non-existing edges given `adj` and `tot_nodes`."""
    true_ids = adj["idx"].t().numpy()
    true_ids = get_edges_ids(true_ids, tot_nodes)

    all_edges_idx = np.arange(tot_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx, all_edges_idx)).reshape(2, -1)

    all_edges_ids = get_edges_ids(all_edges_idx, tot_nodes)

    # only edges that are not in the true_ids should keep here
    mask = np.logical_not(np.isin(all_edges_ids, true_ids))

    non_existing_edges_idx = all_edges_idx[:, mask]
    edges = torch.tensor(non_existing_edges_idx).t()
    vals = torch.zeros(edges.size(0), dtype=torch.long)
    return {"idx": edges, "vals": vals}


def get_non_existing_edges(
    adj,
    number,
    tot_nodes,
    smart_sampling,
    existing_nodes=None,
    forbidden_edges=None,
    forbidden_edge_hashes=None,
    num_nodes_hash_const=None,
):
    """Sample `number` of non-existing edges (negatives) given `adj` and constraints."""
    # Vectorized PyTorch implementation for GPU efficiency
    idx = adj["idx"]
    if idx.shape[1] == 2:
        idx = idx.t()

    # Compute edge hashes using tensor operations
    if num_nodes_hash_const is None:
        num_nodes_hash_const = tot_nodes
    true_hashes = idx[0] * num_nodes_hash_const + idx[1]
    true_hashes_set = set(true_hashes.tolist())

    # Combine forbidden hashes
    if forbidden_edge_hashes is not None:
        forbidden_set = set(forbidden_edge_hashes.tolist())
    elif forbidden_edges is not None:
        forbidden_list = list(forbidden_edges)
        if forbidden_list:
            if isinstance(forbidden_list[0], (list, tuple)):
                forbidden_hashes = [e[0] * num_nodes_hash_const + e[1] for e in forbidden_list]
            else:
                forbidden_hashes = forbidden_list
            forbidden_set = set(forbidden_hashes)
        else:
            forbidden_set = set()
    else:
        forbidden_set = set()

    combined_forbidden = true_hashes_set | forbidden_set
    num_edges = min(number, idx.shape[1] * (idx.shape[1] - 1) - len(true_hashes_set))

    if num_edges <= 0:
        return {
            "idx": torch.empty(0, 2, dtype=torch.long),
            "vals": torch.empty(0, dtype=torch.long),
        }

    # Vectorized batch sampling
    batch_size = max(num_edges * 4, 10000)
    sampled_source = []
    sampled_target = []

    max_iterations = 10
    for _ in range(max_iterations):
        if len(sampled_source) >= num_edges:
            break

        remaining = num_edges - len(sampled_source)
        sample_size = max(remaining * 4, batch_size)

        if smart_sampling and existing_nodes is not None:
            # Sample from existing nodes
            if isinstance(existing_nodes, torch.Tensor):
                existing_np = existing_nodes.cpu().numpy()
            else:
                existing_np = existing_nodes
            source_ids = np.random.choice(idx[0].cpu().numpy(), size=sample_size, replace=True)
            target_ids = np.random.choice(existing_np, size=sample_size, replace=True)
        else:
            # Random sampling
            source_ids = np.random.randint(0, tot_nodes, sample_size)
            target_ids = np.random.randint(0, tot_nodes, sample_size)

        # Filter: no self-loops, not in forbidden set
        for i in range(sample_size):
            if len(sampled_source) >= num_edges:
                break
            if source_ids[i] == target_ids[i]:
                continue
            edge_hash = int(source_ids[i] * num_nodes_hash_const + target_ids[i])
            if edge_hash in combined_forbidden:
                continue
            combined_forbidden.add(edge_hash)
            sampled_source.append(int(source_ids[i]))
            sampled_target.append(int(target_ids[i]))

    if len(sampled_source) == 0:
        return {
            "idx": torch.empty(0, 2, dtype=torch.long),
            "vals": torch.empty(0, dtype=torch.long),
        }

    edges = torch.stack(
        [
            torch.tensor(sampled_source, dtype=torch.long),
            torch.tensor(sampled_target, dtype=torch.long),
        ],
        dim=1,
    )
    vals = torch.zeros(edges.size(0), dtype=torch.long)
    return {"idx": edges, "vals": vals}


def get_edges_ids(sp_idx, tot_nodes):
    """Encode edge pairs `sp_idx` into integer ids using `tot_nodes` as base."""
    return sp_idx[0] * tot_nodes + sp_idx[1]
