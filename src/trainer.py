"""Trainer for GCN-based models on temporal graph tasks.

Handles the training, validation, and evaluation loops for temporal graph learning models.
"""

import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import tqdm

import logger
import utils as u


class Trainer:
    """Trainer for GCN-based models on temporal graph tasks."""

    def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes=2):
        """Initialize the trainer.

        Args:
                args: Configuration namespace.
                splitter: Data splitter for train/dev/test sets.
                gcn: GCN model.
                classifier: Classification head.
                comp_loss: Loss function.
                dataset: Dataset object.
                num_classes: Number of classes.
        """
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier
        self.comp_loss = comp_loss

        self.num_nodes = dataset.num_nodes
        self.data = dataset
        self.num_classes = num_classes

        self.logger = logger.Logger(args, self.num_classes)

        self.init_optimizers(args)

    def init_optimizers(self, args):
        """Initialize optimizers for GCN and classifier.

        Args:
                args: Configuration namespace with learning rate.
        """
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def load_checkpoint(self, filename, model):
        """Load model checkpoint.

        Args:
                filename: Checkpoint file path.
                model: Model object.

        Returns:
                int: Epoch number from checkpoint, or 0 if not found.
        """
        if os.path.isfile(filename):
            print(f"=> loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            epoch = checkpoint["epoch"]
            self.gcn.load_state_dict(checkpoint["gcn_dict"])
            self.classifier.load_state_dict(checkpoint["classifier_dict"])
            self.gcn_opt.load_state_dict(checkpoint["gcn_optimizer"])
            self.classifier_opt.load_state_dict(checkpoint["classifier_optimizer"])
            self.logger.log_str(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
            return epoch
        else:
            self.logger.log_str(f"=> no checkpoint found at '{filename}'")
            return 0

    def train(self):
        """Execute the training loop with validation and early stopping."""
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0

        for e in range(self.args.num_epochs):
            # Set models to training mode
            self.gcn.train()
            self.classifier.train()
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, "TRAIN", grad=True)

            if len(self.splitter.dev) > 0:
                # Set models to evaluation mode
                self.gcn.eval()
                self.classifier.eval()
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, "VALID", grad=False)
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    print("### w" + ") ep " + str(e) + " - Best valid measure:" + str(eval_valid))

            if len(self.splitter.test) > 0:
                # Set models to evaluation mode
                self.gcn.eval()
                self.classifier.eval()
                eval_test, _ = self.run_epoch(self.splitter.test, e, "TEST", grad=False)

                if self.args.save_node_embeddings:
                    log_file = os.path.join(self.args.log_dir, f"epoch_{e}")
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.train_idx, log_file + "_train_nodeembs.csv.gz"
                    )
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.dev_idx, log_file + "_valid_nodeembs.csv.gz"
                    )
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.test_idx, log_file + "_test_nodeembs.csv.gz"
                    )

    def run_epoch(self, split, epoch, set_name, grad):
        """Run a single training/evaluation epoch.

        Args:
                split: Data split (train/dev/test).
                epoch: Epoch number.
                set_name: Name of the split (TRAIN/VALID/TEST).
                grad: Whether to enable gradients.

        Returns:
                tuple: (evaluation metric, node embeddings)
        """
        t0 = time.time()
        log_interval = 999
        if set_name == "TEST":
            log_interval = 1
        self.logger.log_epoch_start(
            epoch, len(split), set_name, minibatch_log_interval=log_interval
        )

        # Use context manager for validation/test to ensure no gradients are computed
        context_manager = torch.no_grad() if not grad else torch.enable_grad()

        with context_manager:
            nodes_embs = None
            # Disable tqdm if console logging is suppressed for this epoch
            disable_pbar = not getattr(self.logger, "console_log", True)
            pbar = tqdm.tqdm(
                split, desc=f"Epoch {epoch} - {set_name}", leave=True, disable=disable_pbar
            )
            running_loss = 0.0
            batch_count = 0

            for s in pbar:
                s = self.prepare_sample(s)

                predictions, nodes_embs = self.predict(
                    s.hist_adj_list, s.hist_ndFeats_list, s.label_sp["idx"], s.node_mask_list
                )

                loss = self.comp_loss(predictions, s.label_sp["vals"])

                # Update running metrics for tqdm
                batch_count += 1
                running_loss = (running_loss * (batch_count - 1) + loss.item()) / batch_count
                pbar.set_postfix(
                    {"loss": f"{running_loss:.4f}", "batch_loss": f"{loss.item():.4f}"}
                )
                self.logger.log_minibatch(
                    predictions, s.label_sp["vals"], loss.detach(), adj=s.label_sp["idx"]
                )

                if grad:
                    self.optim_step(loss)
                    # Force cleanup during training to prevent memory accumulation
                    del predictions, loss
                    if nodes_embs is not None and not self.args.save_node_embeddings:
                        del nodes_embs
                        nodes_embs = None
                    torch.cuda.empty_cache() if self.args.use_cuda else None

                # Free memory for validation/test after each batch
                if not grad:
                    del predictions, loss
                    if nodes_embs is not None and not self.args.save_node_embeddings:
                        del nodes_embs
                        nodes_embs = None
                    torch.cuda.empty_cache() if self.args.use_cuda else None

            # Explicit garbage collection after epoch
            gc.collect()
            if self.args.use_cuda:
                torch.cuda.empty_cache()

        eval_measure = self.logger.log_epoch_done()

        return eval_measure, nodes_embs

    def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, mask_list):
        """Get model predictions for given node pairs.

        Args:
                hist_adj_list: Historical adjacency matrices.
                hist_ndFeats_list: Historical node features.
                node_indices: Indices of node pairs to predict.
                mask_list: Node masks.

        Returns:
                tuple: (predictions, node embeddings)
        """
        nodes_embs = self.gcn(
            hist_adj_list, hist_ndFeats_list, mask_list
        )  # Embedding produced by the GCN

        predict_batch_size = 100000
        gather_predictions = []

        for i in range(1 + (node_indices.size(1) // predict_batch_size)):
            batch_indices = node_indices[:, i * predict_batch_size : (i + 1) * predict_batch_size]
            cls_input = self.gather_node_embs(nodes_embs, batch_indices)
            predictions = self.classifier(cls_input)
            # Detach predictions during inference to avoid keeping computation graph
            if not self.gcn.training:
                predictions = predictions.detach()
            gather_predictions.append(predictions)
            # Free cls_input memory immediately
            del cls_input

        gather_predictions = torch.cat(gather_predictions, dim=0)

        # If not training and embeddings not needed, detach them
        if not self.gcn.training and not self.args.save_node_embeddings:
            nodes_embs = nodes_embs.detach()

        return gather_predictions, nodes_embs

    def gather_node_embs(self, nodes_embs, node_indices):
        """Gather node embeddings for given node indices.

        Args:
                nodes_embs: Node embeddings from GCN.
                node_indices: Indices of nodes to gather.

        Returns:
                Concatenated embeddings for node pairs.
        """
        cls_input = []

        for j, node_set in enumerate(node_indices):
            emb = nodes_embs[node_set]
            cls_input.append(emb)

        # input to classifier is concatenation of node embeddings along feature dimension
        # [emb_u, emb_v]
        return torch.cat(cls_input, dim=1)

    def optim_step(self, loss):
        """Perform an optimization step with gradient accumulation.

        Args:
                loss: Loss value to backpropagate.
        """
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

    def prepare_sample(self, sample):
        """Prepare sample data for model input.

        Args:
                sample: Raw sample dict.

        Returns:
                Processed sample with tensors on correct device.
        """
        sample = SimpleNamespace(**sample)
        for i, adj in enumerate(sample.hist_adj_list):
            # Get node features and handle batch dimension from DataLoader
            node_feats = sample.hist_ndFeats_list[i]

            # DataLoader adds batch dim: [1, num_nodes, features] -> squeeze to [num_nodes, features]
            if node_feats.dim() == 3 and node_feats.size(0) == 1:
                node_feats = node_feats.squeeze(0)

            num_nodes_in_sample = node_feats.shape[0]

            adj = u.sparse_prepare_tensor(adj, torch_size=[num_nodes_in_sample])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = node_feats

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)

            # Handle node mask batch dimension
            node_mask = sample.node_mask_list[i]
            if node_mask.dim() == 2 and node_mask.size(0) == 1:
                node_mask = node_mask.squeeze(0)
            sample.node_mask_list[i] = node_mask.to(
                self.args.device
            ).t()  # transposed to have same dimensions as scorer

        label_sp = {
            "idx": sample.label_sp["idx"][0],
            "vals": sample.label_sp["vals"][0],
        }

        label_sp["idx"] = label_sp["idx"].to(self.args.device).t()

        label_sp["vals"] = label_sp["vals"].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        """Save node embeddings to CSV file.

        Args:
                nodes_embs: Node embeddings.
                indexes: Node indices to save.
                file_name: Output CSV file path.
        """
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.dataset.contID_to_origID[node_id]])

            csv_node_embs.append(
                torch.cat((orig_ID, nodes_embs[node_id].double())).detach().numpy()
            )

        pd.DataFrame(np.array(csv_node_embs)).to_csv(
            file_name, header=None, index=None, compression="gzip"
        )

    def train_incremental(self):
        """Execute incremental training workflow.

        Workflow:
        1. Train on snapshot i, test on snapshot i+1
        2. Fine-tune on snapshot i+1, test on snapshot i+2
        3. Continue until all snapshots are used

        This approach allows the model to adapt to temporal drift in the data.
        """
        # Get train/test pairs - training snapshots use normal negative sampling,
        # test snapshots use all_edges for comprehensive evaluation
        train_test_pairs = []
        for i in range(len(self.splitter.snapshots) - 1):
            # Train on snapshot i (training version), test on snapshot i+1 (test version)
            train_test_pairs.append(
                (self.splitter.snapshots[i], self.splitter._test_snapshots[i + 1])
            )
        num_phases = len(train_test_pairs)

        if num_phases < 1:
            raise ValueError("Incremental training requires at least 2 snapshots")

        all_results = []

        for phase_idx, (train_snapshot, test_snapshot) in enumerate(train_test_pairs):
            # Construct phase description
            # Training accumulates history, so we train on 0...phase_idx
            # And test on phase_idx + 1
            phase_desc = f"Snapshot 0-{phase_idx} -> {phase_idx + 1}"

            print(f"\n{'=' * 60}")
            if phase_idx == 0:
                print(f"{phase_desc} (Initial Training)")
            else:
                print(f"{phase_desc} (Fine-tuning)")
                # Load best checkpoint from previous phase before fine-tuning
                # This ensures we start from the best epoch based on target metric (MAP)
                prev_phase = phase_idx - 1
                best_checkpoint_path = os.path.join(
                    getattr(self.args, "log_dir", "log/"),
                    f"checkpoint_phase_{prev_phase}_best.pth.tar",
                )
                if os.path.isfile(best_checkpoint_path):
                    print(f"\n  Loading best checkpoint from phase {prev_phase}...")
                    checkpoint = torch.load(best_checkpoint_path, map_location=self.args.device)
                    self.gcn.load_state_dict(checkpoint["gcn_dict"])
                    self.classifier.load_state_dict(checkpoint["classifier_dict"])
                    self.gcn_opt.load_state_dict(checkpoint["gcn_optimizer"])
                    self.classifier_opt.load_state_dict(checkpoint["classifier_optimizer"])
                    print(
                        f"  ✓ Loaded from epoch {checkpoint['epoch']} with metric {checkpoint['best_test_metric']:.4f}"
                    )
                else:
                    print(f"  ⚠ Warning: Best checkpoint not found at {best_checkpoint_path}")
                    print(f"     Continuing with current model state...")
            print(f"{'=' * 60}\n")

            phase_results = self._run_incremental_phase(
                train_snapshot=train_snapshot,
                test_snapshot=test_snapshot,
                phase_idx=phase_idx,
                phase_desc=phase_desc,
                is_initial=(phase_idx == 0),
            )

            all_results.append(phase_results)

            # Note: Best checkpoint is saved during training when best metric is achieved
            # See _run_incremental_phase() where checkpoint_phase_{phase_idx}_best.pth.tar is saved

        # Print summary of all phases
        self._print_incremental_summary(all_results)
        return all_results

    def _run_incremental_phase(
        self,
        train_snapshot: torch.utils.data.DataLoader,
        test_snapshot: torch.utils.data.DataLoader,
        phase_idx: int,
        phase_desc: str,
        is_initial: bool,
    ) -> dict:
        """Run a single phase of incremental training.

        Args:
            train_snapshot: DataLoader for training data.
            test_snapshot: DataLoader for test data.
            phase_idx: Index of the current phase.
            phase_desc: Description string for the phase.
            is_initial: Whether this is the initial training phase.

        Returns:
            dict: Results containing train/test metrics and epoch info.
        """
        self.tr_step = 0
        best_eval_train = 0
        best_eval_test = 0
        best_epoch = 0
        epochs_without_impr = 0

        # Use fewer epochs for fine-tuning phases
        num_epochs = self.args.num_epochs if is_initial else self.args.finetune_epochs

        phase_results = {
            "phase_idx": phase_idx,
            "phase_desc": phase_desc,
            "is_initial": is_initial,
            "num_epochs": num_epochs,
            "train_metrics": [],
            "test_metrics": [],
            "best_test_metric": 0,
            "best_epoch": 0,
        }

        # Set phase for logging
        if hasattr(self, "logger") and self.logger:
            self.logger.set_phase(phase_idx, phase_desc)

        for e in range(num_epochs):
            # Training
            self.gcn.train()
            self.classifier.train()
            eval_train, nodes_embs = self.run_epoch(
                train_snapshot, e, f"TRAIN [{phase_desc}]", grad=True
            )
            phase_results["train_metrics"].append(eval_train)

            # Testing
            self.gcn.eval()
            self.classifier.eval()
            eval_test, _ = self.run_epoch(test_snapshot, e, f"TEST [{phase_desc}]", grad=False)
            phase_results["test_metrics"].append(eval_test)

            # Track best performance
            if eval_test > best_eval_test:
                best_eval_test = eval_test
                best_epoch = e
                epochs_without_impr = 0
                target_measure = getattr(self.args, "target_measure", "metric")
                print(f"  -> New best test {target_measure}: {best_eval_test:.4f}")

                # Save checkpoint with the BEST model state (not the last one)
                # This ensures we can resume from the best epoch based on target metric (e.g., MAP)
                log_dir = getattr(self.args, "log_dir", "log/")
                checkpoint_path = os.path.join(
                    log_dir, f"checkpoint_phase_{phase_idx}_best.pth.tar"
                )

                checkpoint_data = {
                    "phase": phase_idx,
                    "phase_desc": phase_desc,
                    "epoch": best_epoch,
                    "best_test_metric": best_eval_test,
                    "target_measure": target_measure,
                    "gcn_dict": self.gcn.state_dict(),
                    "classifier_dict": self.classifier.state_dict(),
                    "gcn_optimizer": self.gcn_opt.state_dict(),
                    "classifier_optimizer": self.classifier_opt.state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"     ✓ Best checkpoint saved (epoch {best_epoch})")
            else:
                epochs_without_impr += 1

        phase_results["best_test_metric"] = best_eval_test
        phase_results["best_epoch"] = best_epoch

        target_measure = getattr(self.args, "target_measure", "metric")
        print(f"\n[{phase_desc}] completed:")
        print(f"  Best test {target_measure}: {best_eval_test:.4f} at epoch {best_epoch}")

        # Compute and save graph structure metrics at the end of the phase (if enabled)
        # This captures the properties of the predicted graph from the last epoch
        if not getattr(self.args, "no_graph_metrics", False):
            print(f"\n  Computing graph structure metrics for {phase_desc}...")
            try:
                # Determine which snapshot we're testing on
                # In incremental training, we test on snapshot (phase_idx + 1)
                test_snapshot_idx = phase_idx + 1

                # Compute graph metrics using accumulated predictions from logger
                # These predictions come from the TEST epoch of the last training epoch
                graph_metrics = self.logger.compute_and_log_phase_graph_metrics(
                    num_nodes=self.num_nodes, phase_idx=phase_idx, snapshot_idx=test_snapshot_idx
                )

                # Add to phase results for checkpoint saving
                phase_results["graph_metrics"] = graph_metrics

                # Save graph files if metrics were computed successfully
                if graph_metrics:
                    self._save_phase_graphs(
                        phase_idx=phase_idx,
                        phase_desc=phase_desc,
                        snapshot_idx=test_snapshot_idx,
                        graph_metrics=graph_metrics,
                    )
                    print(f"  ✓ Graph metrics and files saved successfully")
                else:
                    print(f"  ⚠ No graph metrics computed (no predictions accumulated)")

            except Exception as e:
                print(f"  ⚠ Warning: Could not compute or save graph metrics: {e}")
                phase_results["graph_metrics"] = {}
        else:
            phase_results["graph_metrics"] = {}

        return phase_results

    def _save_phase_graphs(
        self, phase_idx: int, phase_desc: str, snapshot_idx: int, graph_metrics: dict
    ) -> None:
        """Save the predicted graph from this phase in multiple formats.

        This method saves the predicted social network graph at the end of each
        training phase. It creates:
        - .edg file: Simple edge list in X Y format (X follows Y)
        - .graphml file: Rich XML format with probabilities and communities
        - .png file: Visualization of the Giant Connected Component

        Args:
            phase_idx: Current phase index (training on snapshots 0..phase_idx)
            phase_desc: Human-readable description of the phase
            snapshot_idx: Index of the snapshot being tested (phase_idx + 1)
            graph_metrics: Dictionary containing computed graph metrics

        Output Structure:
            graphs/
            └── phase_{phase_idx}_snapshot_{snapshot_idx}/
                ├── predicted_graph.edg          # Edge list X→Y
                ├── predicted_graph.graphml      # XML with metadata
                └── gcc_visualization.png        # GCC plot
        """
        # Create directory structure
        # Using Path for cross-platform compatibility
        graphs_dir = Path("graphs") / f"phase_{phase_idx}_snapshot_{snapshot_idx}"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # Import graph_metrics module
        # Import here to avoid circular imports at module level
        try:
            import graph_metrics as gm
        except ImportError:
            import sys

            sys.path.insert(0, str(Path(__file__).parent))
            import graph_metrics as gm

        # Reconstruct graph from accumulated predictions in logger
        # These predictions were accumulated during the TEST epochs of this phase
        if hasattr(self.logger, "pred_edges_idx") and len(self.logger.pred_edges_idx) > 0:
            # Concatenate all accumulated edge predictions
            # Each minibatch contributed its predictions during log_minibatch()
            all_edge_idx = torch.cat(self.logger.pred_edges_idx, dim=1)  # Shape: [2, total_edges]
            all_edge_vals = torch.cat(self.logger.pred_edges_vals, dim=0)  # Shape: [total_edges]
            all_edge_probs = torch.cat(self.logger.pred_edge_probs, dim=0)  # Shape: [total_edges]

            # Create NetworkX directed graph from predictions
            # This represents the "X follows Y" social network structure
            G = gm.GraphMetricsCalculator.predictions_to_graph(
                all_edge_idx, all_edge_vals, self.num_nodes, threshold=0.5
            )

            # Detect communities using Louvain algorithm
            # Same algorithm and seed (42) as used in compute_all_metrics
            from networkx.algorithms import community as nx_community

            G_undirected = G.to_undirected()
            communities_list = nx_community.louvain_communities(G_undirected, seed=42)

            # Create node-to-community mapping for visualization
            node_community = {}
            for comm_id, comm_nodes in enumerate(communities_list):
                for node in comm_nodes:
                    node_community[node] = comm_id

            # Create edge probability mapping
            # This stores the model's confidence for each predicted link
            edge_prob_dict = {}
            for i in range(all_edge_idx.shape[1]):
                source = int(all_edge_idx[0, i])
                target = int(all_edge_idx[1, i])
                prob = float(all_edge_probs[i])
                edge_prob_dict[(source, target)] = prob

            # Save .edg file (directed: X follows Y)
            # Simple format compatible with existing project .edg files
            edg_path = graphs_dir / "predicted_graph.edg"
            gm.GraphMetricsCalculator.save_graph_edg(G, str(edg_path))
            print(f"    Saved edge list: {edg_path}")

            # Save .graphml with probabilities and communities
            # Rich format for analysis in Gephi, Cytoscape, etc.
            graphml_path = graphs_dir / "predicted_graph.graphml"
            phase_info = {
                "phase": phase_idx,
                "phase_desc": phase_desc,
                "snapshot": snapshot_idx,
                "avg_degree": graph_metrics.get("average_degree", 0),
                "modularity": graph_metrics.get("modularity", 0),
                "num_communities": graph_metrics.get("num_communities", 0),
                "avg_shortest_path": graph_metrics.get("average_shortest_path_length", 0),
                "avg_clustering": graph_metrics.get("average_clustering", 0),
                "gcc_size": graph_metrics.get("gcc_size", 0),
                "total_nodes": graph_metrics.get("total_nodes", 0),
            }
            gm.GraphMetricsCalculator.save_graph_graphml(
                G, edge_prob_dict, node_community, str(graphml_path), phase_info
            )
            print(f"    Saved GraphML: {graphml_path}")

            # Save visualization of GCC (optional, can be disabled with --no-graph-viz)
            # Visual representation with community colors
            viz_path = graphs_dir / "gcc_visualization.png"
            if not getattr(self.args, "no_graph_viz", False):
                title = (
                    f"Phase {phase_idx} - GCC (Snapshot {snapshot_idx})\n"
                    f"{graph_metrics.get('num_communities', 0)} communities, "
                    f"modularity={graph_metrics.get('modularity', 0):.3f}"
                )
                gm.GraphMetricsCalculator.visualize_gcc(G, node_community, str(viz_path), title)
                print(f"    Saved visualization: {viz_path}")
            else:
                print(f"    ⏩ Skipped visualization (disabled with --no-graph-viz)")

            print(f"\n  Graph files saved to: {graphs_dir}")
            print(f"    - {edg_path.name} (edge list)")
            print(f"    - {graphml_path.name} (XML with metadata)")
            if not getattr(self.args, "no_graph_viz", False):
                print(f"    - {viz_path.name} (GCC visualization)")
        else:
            print(f"  ⚠ No predictions accumulated, skipping graph file generation")

    def _print_incremental_summary(self, all_results: list) -> None:
        """Print a summary of all incremental training phases.

        Args:
            all_results: List of results from each phase.
        """
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL INCREMENTAL TRAINING PHASES")
        print("=" * 80)

        # Standard metrics summary
        for result in all_results:
            phase_type = "Initial Training" if result["is_initial"] else "Fine-tuning"
            phase_desc = result.get("phase_desc", f"Phase {result['phase_idx']}")
            print(f"\n{phase_desc} ({phase_type}):")
            print(f"  Best Test Metric: {result['best_test_metric']:.4f}")
            print(f"  Best Epoch: {result['best_epoch']}")
            print(f"  Total Epochs: {len(result['train_metrics'])}")

        # Calculate average performance across all test phases
        test_metrics = [r["best_test_metric"] for r in all_results]
        print(f"\n{'=' * 80}")
        print(f"Overall Test Performance:")
        print(f"  Average: {np.mean(test_metrics):.4f}")
        print(f"  Std Dev: {np.std(test_metrics):.4f}")
        print(f"  Min: {np.min(test_metrics):.4f}")
        print(f"  Max: {np.max(test_metrics):.4f}")

        # NEW: Graph structure metrics summary table
        # Display metrics across all phases for comparison
        phases_with_graph_metrics = [r for r in all_results if r.get("graph_metrics")]

        if phases_with_graph_metrics:
            print(f"\n{'=' * 80}")
            print("GRAPH STRUCTURE METRICS ACROSS PHASES")
            print("=" * 80)
            print("\nThese metrics characterize the predicted social network structure")
            print("at the end of each training phase (computed on TEST snapshot):\n")

            # Create table headers
            headers = [
                "Phase",
                "Snapshot",
                "Avg Degree",
                "Avg Path",
                "Modularity",
                "Avg Cluster",
                "Communities",
                "GCC Ratio",
            ]

            # Build rows for each phase
            rows = []
            for result in phases_with_graph_metrics:
                phase_idx = result["phase_idx"]
                snapshot_idx = phase_idx + 1  # We test on snapshot i+1
                gm = result["graph_metrics"]

                # GCC ratio shows what fraction of nodes are in the main component
                gcc_ratio = gm.get("gcc_size", 0) / gm.get("total_nodes", 1)

                # Format values (handle None for shortest path and modularity)
                avg_path = (
                    f"{gm.get('average_shortest_path_length', 0):.2f}"
                    if gm.get("average_shortest_path_length") is not None
                    else "N/A"
                )
                modularity = (
                    f"{gm.get('modularity', 0):.3f}" if gm.get("modularity") is not None else "N/A"
                )

                rows.append(
                    [
                        phase_idx,
                        snapshot_idx,
                        f"{gm.get('average_degree', 0):.2f}",
                        avg_path,
                        modularity,
                        f"{gm.get('average_clustering', 0):.3f}",
                        gm.get("num_communities", 0),
                        f"{gcc_ratio:.1%}",
                    ]
                )

            # Print simple ASCII table
            col_widths = [
                max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))
            ]

            # Print header
            header_line = " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers))
            print(header_line)
            print("-" * len(header_line))

            # Print rows
            for row in rows:
                row_line = " | ".join(
                    str(cell).center(col_widths[i]) for i, cell in enumerate(row)
                )
                print(row_line)

            # print(f"\nMetrics explanation:")
            # print(f"  • Avg Degree: Average connections per node")
            # print(f"  • Avg Path: Average shortest path length in GCC")
            # print(f"  • Modularity: Community structure strength (0=random, >0.3=communities)")
            # print(f"  • Avg Cluster: Local clustering coefficient (triadic closure)")
            # print(f"  • Communities: Number of detected communities")
            # print(f"  • GCC Ratio: Fraction of nodes in Giant Connected Component")

            print(f"\nGraph files saved in: graphs/phase_{{i}}_snapshot_{{i+1}}/")
            print(
                f"  Each directory contains: .edg (edge list), .graphml (XML), .png (visualization)"
            )
        else:
            print(f"\n{'=' * 80}")
            print("No graph structure metrics available")

        print(f"{'=' * 80}\n")
