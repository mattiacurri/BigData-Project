"""Inference script for synthetic incremental link prediction.

For each batch, computes:
1. New synthetic -> all nodes (real + accumulated synthetic)
2. Real -> new synthetic
3. Old synthetic -> new synthetic (if there are previous synthetics)

All predictions within a batch use the same graph and embeddings.

Functions compute_embeddings(), gather_node_embs() and predict_from_embeddings()
are extracted/adapted from trainer.py for consistency with training.
"""

import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from GabDataset import GabDataset
import modeling.egcn_h as egcn_h
import modeling.MLP as ClassifierHead
import taskers_utils as tu
import utils as u

SOURCE_CHUNK_SIZE = 200
PREDICT_BATCH_SIZE = 300000


class SyntheticDataLoader:
    """Loader for synthetic data batches."""

    def __init__(self, embeddings_path="data/raw/bert_features_synthetic.pkl"):
        """Load synthetic embeddings from `embeddings_path` into memory."""
        with open(embeddings_path, "rb") as f:
            self.synthetic_embeddings = pickle.load(f)

    def load_batch(self, csv_path):
        """Load a synthetic batch CSV."""
        return pd.read_csv(csv_path)

    def compute_user_embeddings(self, df):
        """Compute average embedding for each user based on their posts."""
        user_embs = {}
        for user_id, group in df.groupby("user_id"):
            post_ids = group["post_id"].tolist()
            embeddings = [
                self.synthetic_embeddings[pid]
                for pid in post_ids
                if pid in self.synthetic_embeddings
            ]
            if embeddings:
                user_embs[user_id] = torch.stack(embeddings).mean(dim=0)
        return user_embs

    def get_unique_users(self, df):
        """Get list of unique user IDs in the batch."""
        return df["user_id"].unique().tolist()


def load_trained_model(model_path, device, config=None):
    """Load a trained EGCN-H model and classifier from checkpoint.

    Args:
        model_path: Path to checkpoint file
        device: torch device
        config: Optional config dict with model parameters

    Returns:
        gcn, classifier, args
    """
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if config is not None:
        gcn_params = config.get("gcn_parameters", {})
        args = SimpleNamespace(
            feats_per_node=config.get("gab_args", {}).get("feats_per_node", 768),
            layer_1_feats=gcn_params.get("layer_1_feats", 256),
            layer_2_feats=gcn_params.get("layer_2_feats", 128),
            k_top_grcu=gcn_params.get("k_top_grcu", 768),
            num_layers=gcn_params.get("num_layers", 2),
            lstm_l1_layers=gcn_params.get("lstm_l1_layers", 1),
            lstm_l1_feats=gcn_params.get("lstm_l1_feats", 128),
            lstm_l2_layers=gcn_params.get("lstm_l2_layers", 1),
            lstm_l2_feats=gcn_params.get("lstm_l2_feats", 128),
            gcn_parameters=gcn_params,
        )
    else:
        args = SimpleNamespace(
            feats_per_node=768,
            layer_1_feats=256,
            layer_2_feats=128,
            k_top_grcu=768,
            num_layers=2,
            lstm_l1_layers=1,
            lstm_l1_feats=128,
            lstm_l2_layers=1,
            lstm_l2_feats=128,
            gcn_parameters={"cls_feats": 512},
        )

    gcn = egcn_h.EGCN(args, activation=torch.nn.RReLU(), device=device)
    classifier = ClassifierHead.MLP(args, in_features=args.layer_2_feats * 2)

    gcn.load_state_dict(checkpoint["gcn_dict"])
    classifier.load_state_dict(checkpoint["classifier_dict"])

    gcn = gcn.to(device)
    classifier = classifier.to(device)
    gcn.eval()
    classifier.eval()

    return gcn, classifier, args


def load_real_graph(dataset, max_snapshot):
    """Load real graph edges and node features up to max_snapshot-1.

    Returns:
        adj_dict: Dict with 'idx' and 'vals' for real edges
        features: Tensor [num_nodes, feat_dim]
        num_nodes: Number of real nodes
    """
    hist_snapshot = max_snapshot - 1
    print(
        f"Loading real graph up to snapshot {hist_snapshot} (model trained to predict edges at snapshot {max_snapshot})..."
    )

    adj = tu.get_sp_adj(edges=dataset.edges, time=hist_snapshot, time_window=None)

    features = dataset.get_temporal_node_features(hist_snapshot)

    num_nodes = dataset.num_nodes

    print(f"  Real edges: {adj['idx'].shape[0]}")
    print(f"  Real nodes: {num_nodes}")

    return adj, features, num_nodes


def build_hybrid_adjacency(real_adj, num_real_nodes, total_synthetic_count):
    """Build adjacency matrix with real edges + self-loops for ALL synthetic nodes.

    Args:
        real_adj: Dict with 'idx' and 'vals' for real edges
        num_real_nodes: Number of real nodes
        total_synthetic_count: Total number of synthetic nodes accumulated

    Returns:
        adj_dict: Combined adjacency dict
        total_nodes: Total number of nodes (real + synthetic)
    """
    real_idx = real_adj["idx"]
    real_vals = real_adj["vals"]

    if total_synthetic_count > 0:
        synthetic_ids = torch.arange(num_real_nodes, num_real_nodes + total_synthetic_count)
        self_loops = torch.stack([synthetic_ids, synthetic_ids], dim=1)

        all_idx = torch.cat([real_idx, self_loops], dim=0)
        all_vals = torch.cat([real_vals, torch.ones(total_synthetic_count, dtype=real_vals.dtype)])
    else:
        all_idx = real_idx
        all_vals = real_vals

    return {"idx": all_idx, "vals": all_vals}, num_real_nodes + total_synthetic_count


def create_candidate_edges(source_indices, target_indices):
    """Create candidate edges from source to target (excludes self-loops).

    Args:
        source_indices: List of source node indices
        target_indices: List of target node indices

    Returns:
        Tensor of shape [2, num_candidates]
    """
    candidates = []
    for src in source_indices:
        for dst in target_indices:
            if src != dst:
                candidates.append([src, dst])

    if len(candidates) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(candidates, dtype=torch.long).t()


def prepare_inference_input(adj_dict, features, num_nodes, device):
    """Prepare data in the format expected by compute_embeddings().

    This matches the format used in training (hist_adj_list, hist_ndFeats_list, mask_list).

    Args:
        adj_dict: Dict with 'idx' and 'vals'
        features: Tensor [num_nodes, feat_dim]
        num_nodes: Total number of nodes
        device: torch device

    Returns:
        hist_adj_list: List of adjacency tensors (single element for inference)
        hist_ndFeats_list: List of feature tensors
        mask_list: List of masks
    """
    # IMPORTANT: Compute mask BEFORE normalization!
    # During training, mask is computed on the raw adjacency (without identity matrix).
    # If we compute mask after normalize_adj, all nodes become "active" because
    # normalize_adj adds self-loops (identity) for ALL nodes.
    mask = tu.get_node_mask(adj_dict, num_nodes)

    adj_norm = tu.normalize_adj(adj_dict, num_nodes)

    adj_batched = {
        "idx": adj_norm["idx"].unsqueeze(0),
        "vals": adj_norm["vals"].unsqueeze(0),
    }
    adj_tensor = u.sparse_prepare_tensor(adj_batched, [num_nodes]).to(device)

    hist_adj_list = [adj_tensor]
    hist_ndFeats_list = [features.to(device)]
    mask_list = [mask.to(device)]

    return hist_adj_list, hist_ndFeats_list, mask_list


def gather_node_embs(nodes_embs, node_indices):
    """Gather node embeddings for given node indices.

    Extracted from trainer.py for consistency with training.

    Args:
        nodes_embs: Node embeddings from GCN [num_nodes, emb_dim]
        node_indices: Tensor [2, num_pairs] - indices of node pairs

    Returns:
        Concatenated embeddings for node pairs [num_pairs, 2*emb_dim]
    """
    if node_indices.shape[1] == 0:
        return torch.zeros((0, nodes_embs.size(1) * 2), device=nodes_embs.device)

    max_idx = node_indices.max().item()
    if max_idx >= nodes_embs.size(0):
        raise IndexError(
            f"Node index {max_idx} out of bounds for embedding tensor with size {nodes_embs.size(0)}"
        )

    cls_input = []
    for j, node_set in enumerate(node_indices):
        emb = nodes_embs[node_set]
        cls_input.append(emb)

    return torch.cat(cls_input, dim=1)


def compute_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list):
    """Compute node embeddings from the GCN.

    Args:
        gcn: EGCN-H model
        hist_adj_list: List of adjacency tensors (prepared with sparse_prepare_tensor)
        hist_ndFeats_list: List of node feature tensors
        mask_list: List of node masks

    Returns:
        Node embeddings tensor [num_nodes, emb_dim]
    """
    with torch.inference_mode():
        nodes_embs = gcn(hist_adj_list, hist_ndFeats_list, mask_list)
    return nodes_embs.detach()


def predict_from_embeddings(classifier, node_embs, candidates, desc="Predicting"):
    """Predict probabilities for candidate edges using pre-computed embeddings.

    Uses gather_node_embs for consistency with training.

    Args:
        classifier: MLP classifier
        node_embs: Pre-computed node embeddings [num_nodes, emb_dim]
        candidates: Candidate edges tensor [2, num_candidates]
        desc: Description for progress bar

    Returns:
        Tensor of probabilities for each candidate edge
    """
    if candidates.shape[1] == 0:
        return torch.zeros(0)

    predictions = []
    num_batches = 1 + (candidates.size(1) // PREDICT_BATCH_SIZE)

    with torch.inference_mode():
        for i in tqdm(range(num_batches), total=num_batches, desc=desc, leave=False):
            batch = candidates[:, i * PREDICT_BATCH_SIZE : (i + 1) * PREDICT_BATCH_SIZE]
            cls_input = gather_node_embs(node_embs, batch)
            pred = classifier(cls_input)
            predictions.append(pred.cpu())
            del cls_input

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_preds = torch.cat(predictions, dim=0)
    probs = torch.softmax(all_preds, dim=1)[:, 1]
    return probs


def save_predictions(candidates, probs, threshold, all_probabilities, all_predicted_edges):
    """Save predictions to dictionaries.

    Args:
        candidates: Candidate edges tensor [2, num_candidates]
        probs: Probabilities tensor
        threshold: Threshold for positive edges
        all_probabilities: Dict to store all probabilities
        all_predicted_edges: List to store positive edges

    Returns:
        Number of positive edges
    """
    positive_count = 0
    for i in range(candidates.shape[1]):
        src = candidates[0, i].item()
        dst = candidates[1, i].item()
        prob = probs[i].item()

        all_probabilities[(src, dst)] = prob

        if prob >= threshold:
            all_predicted_edges.append((src, dst, prob))
            positive_count += 1

    return positive_count


def save_predictions_with_categories(
    candidates,
    probs,
    threshold,
    all_probabilities,
    all_predicted_edges,
    num_real_nodes,
    new_synth_start_idx,
    new_synth_end_idx,
    batch_stats_detail,
    category_prefix,
):
    """Save predictions and categorize them by edge type.

    Categories for new_synth sources:
        - new_synth_to_real: new synth -> real nodes
        - new_synth_to_old_synth: new synth -> old synth nodes
        - new_synth_to_new_synth: new synth -> new synth (same batch)

    Args:
        candidates: Candidate edges tensor [2, num_candidates]
        probs: Probabilities tensor
        threshold: Threshold for positive edges
        all_probabilities: Dict to store all probabilities
        all_predicted_edges: List to store positive edges
        num_real_nodes: Number of real nodes
        new_synth_start_idx: Start index of new synthetic nodes in this batch
        new_synth_end_idx: End index of new synthetic nodes in this batch
        batch_stats_detail: Dict to update with categorized stats
        category_prefix: Prefix for category names in batch_stats_detail

    Returns:
        Total positive count
    """
    categories = {
        "new_synth_to_real": {"predictions": 0, "positive": 0},
        "new_synth_to_old_synth": {"predictions": 0, "positive": 0},
        "new_synth_to_new_synth": {"predictions": 0, "positive": 0},
    }

    for i in range(candidates.shape[1]):
        src = candidates[0, i].item()
        dst = candidates[1, i].item()
        prob = probs[i].item()

        all_probabilities[(src, dst)] = prob

        if dst < num_real_nodes:
            cat = "new_synth_to_real"
        elif dst >= new_synth_start_idx and dst < new_synth_end_idx:
            cat = "new_synth_to_new_synth"
        else:
            cat = "new_synth_to_old_synth"

        categories[cat]["predictions"] += 1
        if prob >= threshold:
            all_predicted_edges.append((src, dst, prob))
            categories[cat]["positive"] += 1

    for cat, stats in categories.items():
        if cat not in batch_stats_detail:
            batch_stats_detail[cat] = {"predictions": 0, "positive": 0}
        batch_stats_detail[cat]["predictions"] += stats["predictions"]
        batch_stats_detail[cat]["positive"] += stats["positive"]

    return sum(s["positive"] for s in categories.values())


def main():
    """Run incremental inference on synthetic batches using a trained EGCN-H model."""
    parser = argparse.ArgumentParser(
        description="Inference for synthetic incremental link prediction"
    )
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument("--model_path", help="Path to trained model checkpoint")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for positive link prediction"
    )
    parser.add_argument("--output_dir", default="synthetic_results", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=SOURCE_CHUNK_SIZE,
        help="Number of source nodes to process at once",
    )
    args = parser.parse_args()

    config = None
    if args.config:
        print(f"Loading config from {args.config}...")
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        if not args.model_path:
            args.model_path = config.get(
                "initial_checkpoint", "log/checkpoint_phase_4_best.pth.tar"
            )

    device = args.device
    print(f"Using device: {device}")
    chunk_size = args.chunk_size

    print(f"\n{'=' * 60}")
    print(f"Loading model from {args.model_path}")
    print(f"{'=' * 60}")
    gcn, classifier, model_args = load_trained_model(args.model_path, device, config)

    print(f"\n{'=' * 60}")
    print("Loading real dataset")
    print(f"{'=' * 60}")
    real_args = SimpleNamespace(gab_args={"folder": "./data/", "feats_per_node": 768})
    real_dataset = GabDataset(real_args)

    max_snapshot = int(real_dataset.max_time.item())
    # print(f"Max snapshot: {max_snapshot}")

    real_adj, real_features, num_real_nodes = load_real_graph(real_dataset, max_snapshot)

    synthetic_data = SyntheticDataLoader()

    all_predicted_edges = []
    all_probabilities = {}
    all_synthetic_users = {}
    all_user_embeddings = {}
    batch_stats = {}

    cumulative_synth_features = []
    total_synthetic_count = 0
    previous_synth_count = 0

    for batch_num in tqdm([1, 2, 3], desc="Processing batches"):
        batch_csv = f"data/raw/batch{batch_num}_synthetic.csv"
        batch_df = synthetic_data.load_batch(batch_csv)
        user_embs = synthetic_data.compute_user_embeddings(batch_df)
        new_users = synthetic_data.get_unique_users(batch_df)

        # print(f" New users in batch: {len(new_users)}")
        # print(f"Users with embeddings: {len(user_embs)}")

        all_user_embeddings.update(user_embs)

        batch_features = []
        for user in new_users:
            emb = user_embs.get(user, torch.zeros(768))
            batch_features.append(emb)

        cumulative_synth_features.extend(batch_features)
        batch_size = len(new_users)

        batch_start_idx = num_real_nodes + total_synthetic_count
        total_synthetic_count += batch_size

        all_synth_feats = torch.stack(cumulative_synth_features)
        feat_matrix = torch.cat([real_features, all_synth_feats], dim=0)

        hybrid_adj, total_nodes = build_hybrid_adjacency(
            real_adj, num_real_nodes, total_synthetic_count
        )

        # print(
        #     f"Graph built: {total_nodes} nodes ({num_real_nodes} real + {total_synthetic_count} synthetic)"
        # )
        # print(f"Preparing inference input...")

        hist_adj_list, hist_ndFeats_list, mask_list = prepare_inference_input(
            hybrid_adj, feat_matrix, total_nodes, device
        )

        print(f"Computing embeddings...")

        node_embs = compute_embeddings(gcn, hist_adj_list, hist_ndFeats_list, mask_list)

        new_synth_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
        all_node_indices = list(range(total_nodes))
        real_indices = list(range(num_real_nodes))
        old_synth_indices = (
            list(range(num_real_nodes, batch_start_idx)) if previous_synth_count > 0 else []
        )

        batch_stats_detail = {
            "new_synth_to_real": {"predictions": 0, "positive": 0},
            "new_synth_to_old_synth": {"predictions": 0, "positive": 0},
            "new_synth_to_new_synth": {"predictions": 0, "positive": 0},
            "real_to_new_synth": {"predictions": 0, "positive": 0},
            "old_synth_to_new_synth": {"predictions": 0, "positive": 0},
        }

        new_synth_end_idx = batch_start_idx + batch_size

        # print(f"\n--- Type 1: New synthetic -> all nodes ---")
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        for chunk_idx in tqdm(range(num_chunks), desc="New->All chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_size_actual = chunk_end - chunk_start

            source_indices = [batch_start_idx + i for i in range(chunk_start, chunk_end)]
            candidates = create_candidate_edges(source_indices, all_node_indices)

            # print(
            #     f"  Chunk {chunk_start // chunk_size + 1}: {chunk_size_actual} sources, {candidates.shape[1]} candidates"
            # )

            if candidates.shape[1] > 0:
                probs = predict_from_embeddings(
                    classifier,
                    node_embs,
                    candidates.to(device),
                    desc=f"  New->All {chunk_idx + 1}",
                )
                save_predictions_with_categories(
                    candidates,
                    probs,
                    args.threshold,
                    all_probabilities,
                    all_predicted_edges,
                    num_real_nodes,
                    batch_start_idx,
                    new_synth_end_idx,
                    batch_stats_detail,
                    "new_synth",
                )
                del probs

            del candidates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # print(f"\n--- Type 2: Real -> new synthetic ---")
        num_chunks = (num_real_nodes + chunk_size - 1) // chunk_size
        for chunk_idx in tqdm(range(num_chunks), desc="Real->Synth chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_real_nodes)
            chunk_size_actual = chunk_end - chunk_start

            source_indices = list(range(chunk_start, chunk_end))
            candidates = create_candidate_edges(source_indices, new_synth_indices)

            # print(
            #     f"  Chunk {chunk_idx + 1}: {chunk_size_actual} sources, {candidates.shape[1]} candidates"
            # )

            if candidates.shape[1] > 0:
                probs = predict_from_embeddings(
                    classifier,
                    node_embs,
                    candidates.to(device),
                    desc=f"  Real->Synth {chunk_idx + 1}",
                )
                positive_count = save_predictions(
                    candidates, probs, args.threshold, all_probabilities, all_predicted_edges
                )
                batch_stats_detail["real_to_new_synth"]["predictions"] += candidates.shape[1]
                batch_stats_detail["real_to_new_synth"]["positive"] += positive_count
                # print(f"    Positive: {positive_count}")
                del probs

            del candidates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(old_synth_indices) > 0:
            # print(
            #     f"\n--- Type 3: Old synthetic -> new synthetic ({len(old_synth_indices)} old) ---"
            # )
            num_chunks = (len(old_synth_indices) + chunk_size - 1) // chunk_size
            for chunk_idx in tqdm(range(num_chunks), desc="Old->New chunks"):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(old_synth_indices))
                chunk_size_actual = chunk_end - chunk_start

                source_indices = old_synth_indices[chunk_start:chunk_end]
                candidates = create_candidate_edges(source_indices, new_synth_indices)

                # print(
                #     f"  Chunk {chunk_idx + 1}: {chunk_size_actual} sources, {candidates.shape[1]} candidates"
                # )

                if candidates.shape[1] > 0:
                    probs = predict_from_embeddings(
                        classifier,
                        node_embs,
                        candidates.to(device),
                        desc=f"  Old->New {chunk_idx + 1}",
                    )
                    positive_count = save_predictions(
                        candidates, probs, args.threshold, all_probabilities, all_predicted_edges
                    )
                    batch_stats_detail["old_synth_to_new_synth"]["predictions"] += (
                        candidates.shape[1]
                    )
                    batch_stats_detail["old_synth_to_new_synth"]["positive"] += positive_count
                    # print(f"    Positive: {positive_count}")
                    del probs

                del candidates
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # else:
        # print(f"\n--- Type 3: Old synthetic -> new synthetic (skipped, no old synthetics) ---")

        for i, user in tqdm(
            enumerate(new_users), total=len(new_users), desc="Mapping new users to indices"
        ):
            node_idx = batch_start_idx + i
            all_synthetic_users[user] = node_idx

        total_predictions = (
            batch_stats_detail["new_synth_to_real"]["predictions"]
            + batch_stats_detail["new_synth_to_old_synth"]["predictions"]
            + batch_stats_detail["new_synth_to_new_synth"]["predictions"]
            + batch_stats_detail["real_to_new_synth"]["predictions"]
            + batch_stats_detail["old_synth_to_new_synth"]["predictions"]
        )
        total_positive = (
            batch_stats_detail["new_synth_to_real"]["positive"]
            + batch_stats_detail["new_synth_to_old_synth"]["positive"]
            + batch_stats_detail["new_synth_to_new_synth"]["positive"]
            + batch_stats_detail["real_to_new_synth"]["positive"]
            + batch_stats_detail["old_synth_to_new_synth"]["positive"]
        )

        batch_stats[batch_num] = {
            "new_users": len(new_users),
            "previous_synth_count": previous_synth_count,
            "total_synth_count": total_synthetic_count,
            "predictions": batch_stats_detail,
            "total_predictions": total_predictions,
            "total_positive": total_positive,
        }

        print(f"\nBatch {batch_num} summary:")
        print(
            f"  New->Real:     {batch_stats_detail['new_synth_to_real']['predictions']:,} pred, {batch_stats_detail['new_synth_to_real']['positive']:,} pos"
        )
        print(
            f"  New->OldSynth: {batch_stats_detail['new_synth_to_old_synth']['predictions']:,} pred, {batch_stats_detail['new_synth_to_old_synth']['positive']:,} pos"
        )
        print(
            f"  New->NewSynth: {batch_stats_detail['new_synth_to_new_synth']['predictions']:,} pred, {batch_stats_detail['new_synth_to_new_synth']['positive']:,} pos"
        )
        print(
            f"  Real->New:     {batch_stats_detail['real_to_new_synth']['predictions']:,} pred, {batch_stats_detail['real_to_new_synth']['positive']:,} pos"
        )
        print(
            f"  Old->New:      {batch_stats_detail['old_synth_to_new_synth']['predictions']:,} pred, {batch_stats_detail['old_synth_to_new_synth']['positive']:,} pos"
        )
        print(f"  TOTAL:         {total_predictions:,} predictions, {total_positive:,} positive")

        del node_embs, feat_matrix, hist_adj_list, hist_ndFeats_list, mask_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        previous_synth_count = total_synthetic_count

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Saving results to {output_dir}")
    print(f"{'=' * 60}")

    with open(output_dir / "complete_graph.edg", "w") as f:
        for src, dst, prob in sorted(all_predicted_edges, key=lambda x: -x[2]):
            f.write(f"{src} {dst} {prob:.6f}\n")
    print(f"Saved complete_graph.edg with {len(all_predicted_edges)} edges")

    synth_node_set = set(all_synthetic_users.values())
    synth_edges = [
        (s, d, p) for s, d, p in all_predicted_edges if s in synth_node_set and d in synth_node_set
    ]

    with open(output_dir / "synthetic_only.edg", "w") as f:
        for src, dst, prob in sorted(synth_edges, key=lambda x: -x[2]):
            f.write(f"{src} {dst} {prob:.6f}\n")
    print(f"Saved synthetic_only.edg with {len(synth_edges)} edges")

    real_to_synth_edges = [
        (s, d, p) for s, d, p in all_predicted_edges if s < num_real_nodes and d in synth_node_set
    ]

    with open(output_dir / "real_to_synthetic.edg", "w") as f:
        for src, dst, prob in sorted(real_to_synth_edges, key=lambda x: -x[2]):
            f.write(f"{src} {dst} {prob:.6f}\n")
    print(f"Saved real_to_synthetic.edg with {len(real_to_synth_edges)} edges")

    with open(output_dir / "full_predictions.pkl", "wb") as f:
        pickle.dump(all_probabilities, f)
    print(f"Saved full_predictions.pkl with {len(all_probabilities)} predictions")

    summary = {
        "real_nodes": num_real_nodes,
        "synthetic_users": len(all_synthetic_users),
        "total_positive_edges": len(all_predicted_edges),
        "synthetic_only_edges": len(synth_edges),
        "real_to_synthetic_edges": len(real_to_synth_edges),
        "total_predictions": len(all_probabilities),
        "threshold": args.threshold,
        "model_path": args.model_path,
        "chunk_size": chunk_size,
        "batch_stats": batch_stats,
    }

    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved inference_summary.json")

    print(f"\n{'=' * 60}")
    print("INFERENCE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total synthetic users processed: {len(all_synthetic_users)}")
    print(f"Total predictions made: {len(all_probabilities)}")
    print(f"Total positive edges (prob >= {args.threshold}): {len(all_predicted_edges)}")
    print(f"  Synthetic -> Synthetic: {len(synth_edges)}")
    print(f"  Real -> Synthetic: {len(real_to_synth_edges)}")


if __name__ == "__main__":
    main()
