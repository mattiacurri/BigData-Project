"""Trainer for GCN-based models on temporal graph tasks.

Handles the training, validation, and evaluation loops for temporal graph learning models.
"""

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
        self.gcn_opt = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay if hasattr(args, "weight_decay") else 0,
        )
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay if hasattr(args, "weight_decay") else 0,
        )
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def load_checkpoint(self, filename: str, load_optimizer_state: bool = True) -> dict:
        """Load model checkpoint.

        Args:
            filename: Checkpoint file path.
            load_optimizer_state: If True, load optimizer state. If False, only load model weights.
                                  Setting to False enables classic fine-tuning with fresh optimizer state.

        Returns:
            dict: The loaded checkpoint dictionary, or empty dict if not found.
        """
        if os.path.isfile(filename):
            print(f"=> loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location=self.args.device)
            self.gcn.load_state_dict(checkpoint["gcn_dict"])
            self.classifier.load_state_dict(checkpoint["classifier_dict"])

            if load_optimizer_state:
                self.gcn_opt.load_state_dict(checkpoint["gcn_optimizer"])
                self.classifier_opt.load_state_dict(checkpoint["classifier_optimizer"])
                optimizer_status = "with optimizer state"
            else:
                # Reinitialize optimizers for classic fine-tuning (fresh momentum/adaptive rates)
                self.init_optimizers(self.args)
                optimizer_status = "with RESET optimizer (classic fine-tuning)"

            msg = f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']}) {optimizer_status}"
            if "best_test_metric" in checkpoint:
                msg += f" | metric {checkpoint['best_test_metric']:.4f}"
            self.logger.log_str(msg)
            return checkpoint
        self.logger.log_str(f"=> no checkpoint found at '{filename}'")
        return {}

    def train(self):
        """Execute the training loop with validation and early stopping."""
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0

        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

        for e in range(self.args.num_epochs):
            # Set models to training mode
            self.gcn.train()
            self.classifier.train()
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, "TRAIN", grad=True)
            if len(self.splitter.dev) > 0:
                # Set models to evaluation mode
                self.gcn.eval()
                self.classifier.eval()
                with torch.no_grad():
                    eval_valid, _ = self.run_epoch(self.splitter.dev, e, "VALID", grad=False)
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    print("### w" + ") ep " + str(e) + " - Best valid measure:" + str(eval_valid))

            if len(self.splitter.test) > 0:
                # Set models to evaluation mode
                self.gcn.eval()
                self.classifier.eval()
                with torch.no_grad():
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
        log_interval = 1 if set_name == "TEST" else 999
        self.logger.log_epoch_start(
            epoch, len(split), set_name, minibatch_log_interval=log_interval
        )

        nodes_embs = None
        running_loss = 0.0

        pbar = tqdm(split, desc=f"Epoch {epoch} - {set_name}")

        for i, s in enumerate(pbar):
            s = self.prepare_sample(s)

            predictions, nodes_embs = self.predict(
                s.hist_adj_list, s.hist_ndFeats_list, s.label_sp["idx"], s.node_mask_list
            )

            loss = (
                None
                if set_name.startswith("TEST")
                else self.comp_loss(predictions, s.label_sp["vals"])
            )

            if grad:
                self.optim_step(loss)

            # Update metrics and progress bar
            if loss is not None:
                running_loss = (running_loss * i + loss.item()) / (i + 1)
                pbar.set_postfix(loss=f"{running_loss:.4f}")

            self.logger.log_minibatch(predictions, s.label_sp["vals"], loss, adj=s.label_sp["idx"])

            # Force cleanup to prevent memory accumulation
            del predictions
            if loss is not None:
                del loss
            if nodes_embs is not None and not self.args.save_node_embeddings:
                nodes_embs = None

            if self.args.use_cuda:
                torch.cuda.empty_cache()

        # Explicit garbage collection after epoch
        gc.collect()
        if self.args.use_cuda:
            torch.cuda.empty_cache()

        return self.logger.log_epoch_done(), nodes_embs

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

        # Safety check: verify indices are within bounds
        max_idx = node_indices.max().item()
        if max_idx >= nodes_embs.size(0):
            raise IndexError(
                f"Node index {max_idx} out of bounds for embedding tensor with size {nodes_embs.size(0)}.\n"
                f"This usually means label edges were not properly remapped to the compacted node space.\n"
                f"node_indices shape: {node_indices.shape}, nodes_embs shape: {nodes_embs.shape}"
            )

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
        for i in range(len(self.splitter.train_snapshots) - 1):
            # Train on snapshot i (training version), test on snapshot i+1 (test version)
            # This ensures we predict the NEXT time step (future prediction)
            train_test_pairs.append(
                (self.splitter.train_snapshots[i], self.splitter.test_snapshots[i + 1])
            )

        if len(train_test_pairs) < 1:
            raise ValueError("Incremental training requires at least 2 snapshots")

        all_results = []

        for phase_idx, (train_snapshot, test_snapshot) in enumerate(train_test_pairs):
            # Construct phase description
            # train_snapshots[i] corresponds to time step (i+1) due to history requirements
            # We train on time step (phase_idx + 1) and test on (phase_idx + 2)
            train_time_step = phase_idx + 1
            test_time_step = phase_idx + 2
            phase_desc = f"Train on snapshot {train_time_step}, test on snapshot {test_time_step}"

            print(f"\n{'=' * 60}")
            print(f"{phase_desc} Training")
            # Load best checkpoint from previous phase before fine-tuning
            # This ensures we start from the best epoch based on target metric
            prev_phase = phase_idx - 1
            best_checkpoint_path = os.path.join(
                getattr(self.args, "log_dir", "log/"),
                f"checkpoint_phase_{prev_phase}_best.pth.tar",
            )
            if os.path.isfile(best_checkpoint_path):
                print(f"\n  Loading best checkpoint from phase {prev_phase}...")
                # Use reset_optimizer_on_finetune flag to control optimizer state loading
                # True = classic fine-tuning (fresh optimizer), False = continue training (keep momentum)
                reset_optimizer = getattr(self.args, "reset_optimizer_on_finetune", True)
                self.load_checkpoint(
                    best_checkpoint_path, load_optimizer_state=not reset_optimizer
                )
            else:
                print(f"  Warning: Best checkpoint not found at {best_checkpoint_path}")
                print("  Continuing with current model state...")
            print(f"{'=' * 60}\n")

            phase_results = self._run_incremental_phase(
                train_snapshot=train_snapshot,
                test_snapshot=test_snapshot,
                phase_idx=phase_idx,
                phase_desc=phase_desc,
                is_initial=(phase_idx == 0),
            )

            all_results.append(phase_results)

        # Print summary of all phases
        self._print_incremental_summary(all_results)

        # Save final predictions from the best model of the last phase
        self._save_final_predictions(all_results)

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
        best_eval_test = 0
        best_epoch = 0
        epochs_without_impr = 0

        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

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
            with torch.no_grad():
                eval_test, _ = self.run_epoch(test_snapshot, e, f"TEST [{phase_desc}]", grad=False)
            phase_results["test_metrics"].append(eval_test)

            # Track best performance
            if eval_test > best_eval_test:
                best_eval_test = eval_test
                best_epoch = e
                epochs_without_impr = 0
                target_measure = getattr(self.args, "target_measure", "metric")
                print(
                    f"  -> New best test {target_measure}: {best_eval_test:.4f} (epoch {best_epoch}); saving checkpoint..."
                )

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
            else:
                epochs_without_impr += 1

        phase_results["best_test_metric"] = best_eval_test
        phase_results["best_epoch"] = best_epoch

        target_measure = getattr(self.args, "target_measure", "metric")
        print(f"\n[{phase_desc}] completed:")
        print(f"  Best test {target_measure}: {best_eval_test:.4f} at epoch {best_epoch}")

        return phase_results

    def _save_phase_graphs(
        self, phase_idx: int, phase_desc: str, snapshot_idx: int, graph_metrics: dict
    ) -> None:
        """Save the predicted graph from this phase in multiple formats.

        This method saves the predicted social network graph at the end of each
        training phase. It creates:
        - .edg file: Simple edge list in X Y format (X follows Y)
        - .png file: Visualization of the Giant Connected Component

        Args:
            phase_idx: Current phase index (0-based)
            phase_desc: Human-readable description of the phase
            snapshot_idx: Index of the snapshot being tested (phase_idx + 2)
            graph_metrics: Dictionary containing computed graph metrics

        Output Structure:
            graphs/
            └── train_snapshot_{train}_test_snapshot_{test}/
                ├── predicted_graph.edg          # Edge list X→Y
                └── gcc_visualization.png        # GCC plot
        """
        # Create directory structure with clear train/test snapshot names
        # Using Path for cross-platform compatibility
        train_snapshot_idx = phase_idx + 1
        graphs_dir = (
            Path("graphs") / f"train_snapshot_{train_snapshot_idx}_test_snapshot_{snapshot_idx}"
        )
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

            print(f"\n  Graph files saved to: {graphs_dir}")
            print(f"    - {edg_path.name} (edge list)")
            if not getattr(self.args, "no_graph_viz", False):
                print(f"    - {viz_path.name} (GCC visualization)")
        else:
            print("  ⚠ No predictions accumulated, skipping graph file generation")

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
        print("Overall Test Performance:")
        print(f"  Average: {np.mean(test_metrics):.4f}")
        print(f"  Std Dev: {np.std(test_metrics):.4f}")
        print(f"  Min: {np.min(test_metrics):.4f}")
        print(f"  Max: {np.max(test_metrics):.4f}")
        print(f"{'=' * 80}\n")

    def _save_final_predictions(self, all_results: list) -> None:
        """Save final predictions from the best model of the last phase.

        This method loads the best checkpoint from the last training phase,
        runs inference on the final test snapshot, and saves all predictions
        with their probabilities in .edg format (source target probability).

        Args:
            all_results: List of results from all phases.
        """
        if not all_results:
            print("⚠ No results available, skipping final predictions save")
            return

        # Get the last phase
        last_phase = all_results[-1]
        phase_idx = last_phase["phase_idx"]

        print(f"\n{'=' * 80}")
        print("SAVING FINAL PREDICTIONS")
        print("=" * 80)

        # Load best checkpoint from last phase
        log_dir = getattr(self.args, "log_dir", "log/")
        checkpoint_path = os.path.join(log_dir, f"checkpoint_phase_{phase_idx}_best.pth.tar")

        if not os.path.isfile(checkpoint_path):
            print(f"⚠ Best checkpoint not found at {checkpoint_path}")
            print("  Skipping final predictions save")
            return

        print(f"\n  Loading best checkpoint from phase {phase_idx}...")
        self.load_checkpoint(checkpoint_path, load_optimizer_state=False)

        # Get the test snapshot for the last phase
        # train_snapshots[phase_idx] corresponds to time step (phase_idx + 1)
        # We test on time step (phase_idx + 2)
        test_snapshot_idx = phase_idx + 1
        if test_snapshot_idx >= len(self.splitter.test_snapshots):
            print(f"⚠ Test snapshot index {test_snapshot_idx} out of range")
            return

        test_snapshot = self.splitter.test_snapshots[test_snapshot_idx]

        print("  Running inference on final test snapshot...")

        # Set models to evaluation mode
        self.gcn.eval()
        self.classifier.eval()

        # Run inference and collect all predictions
        all_predictions = []
        all_edge_indices = []
        all_true_labels = []

        with torch.no_grad():
            for i, sample in enumerate(test_snapshot):
                sample = self.prepare_sample(sample)

                # Get predictions
                predictions, _ = self.predict(
                    sample.hist_adj_list,
                    sample.hist_ndFeats_list,
                    sample.label_sp["idx"],
                    sample.node_mask_list,
                )

                # Get probabilities for positive class
                probs = torch.softmax(predictions, dim=1)[:, 1]

                # Get edge indices and true labels
                edge_idx = sample.label_sp["idx"]  # Shape: [2, num_edges]
                true_labels = sample.label_sp["vals"]  # Shape: [num_edges]

                # Store results
                all_predictions.append(probs.cpu())
                all_edge_indices.append(edge_idx.cpu())
                all_true_labels.append(true_labels.cpu())

        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_edge_indices = torch.cat(all_edge_indices, dim=1)  # [2, total_edges]
        all_true_labels = torch.cat(all_true_labels, dim=0)

        # Create output directory
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions in .edg format with 3 columns: source target probability
        edg_path = predictions_dir / "final_predictions.edg"
        with open(edg_path, "w") as f:
            for i in range(all_edge_indices.shape[1]):
                source = int(all_edge_indices[0, i])
                target = int(all_edge_indices[1, i])
                prob = float(all_predictions[i])
                f.write(f"{source} {target} {prob:.6f}\n")

        print(f"\n  ✓ Saved predictions to: {edg_path}")
        print(f"    Total predictions: {all_edge_indices.shape[1]}")
        print(f"    Positive edges: {all_true_labels.sum().item()}")
        print(f"    Negative edges: {(all_true_labels == 0).sum().item()}")

        # Build complete graph: historical edges + predicted positive edges
        print("\n  Building complete graph with historical + predicted edges...")

        # Get training snapshot for historical edges
        train_snapshot_idx = phase_idx
        train_snapshot = self.splitter.train_snapshots[train_snapshot_idx]

        # Collect historical edges from training
        historical_edges = set()
        with torch.no_grad():
            for sample in train_snapshot:
                sample = self.prepare_sample(sample)
                # Get predictions on training to identify historical structure
                predictions_hist, _ = self.predict(
                    sample.hist_adj_list,
                    sample.hist_ndFeats_list,
                    sample.label_sp["idx"],
                    sample.node_mask_list,
                )
                edge_idx_hist = sample.label_sp["idx"]
                true_labels_hist = sample.label_sp["vals"]

                # Add edges that are truly positive (historical ground truth)
                for i in range(edge_idx_hist.shape[1]):
                    if true_labels_hist[i] == 1:
                        source = int(edge_idx_hist[0, i])
                        target = int(edge_idx_hist[1, i])
                        historical_edges.add((source, target))

        # Get predicted positive edges from test
        predicted_edges = []
        for i in range(all_edge_indices.shape[1]):
            if all_predictions[i] >= 0.5:  # Predicted as positive
                source = int(all_edge_indices[0, i])
                target = int(all_edge_indices[1, i])
                prob = float(all_predictions[i])
                predicted_edges.append((source, target, prob))

        # Combine: historical + predicted
        complete_graph_edges = historical_edges.copy()
        new_predicted_edges = []
        for source, target, prob in predicted_edges:
            if (source, target) not in complete_graph_edges:
                complete_graph_edges.add((source, target))
                new_predicted_edges.append((source, target, prob))

        # Save complete graph
        complete_edg_path = predictions_dir / "complete_graph_predicted.edg"
        with open(complete_edg_path, "w") as f:
            # Write historical edges (without probability, they're known)
            for source, target in sorted(historical_edges):
                f.write(f"{source} {target} 1.000000\n")
            # Write predicted edges (with their probability)
            for source, target, prob in sorted(new_predicted_edges):
                f.write(f"{source} {target} {prob:.6f}\n")

        print(f"  ✓ Saved complete graph to: {complete_edg_path}")
        print(f"    Historical edges: {len(historical_edges)}")
        print(f"    Predicted positive edges: {len(new_predicted_edges)}")
        print(f"    Total edges in complete graph: {len(complete_graph_edges)}")
        print(f"\n{'=' * 80}\n")
