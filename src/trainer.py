"""Trainer for GCN-based models on temporal graph tasks.

Handles the training, validation, and evaluation loops for temporal graph learning models.
"""

import gc
import os
import time

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
        epochs_without_impr = 0

        for e in range(self.args.num_epochs):
            # Set models to training mode
            self.gcn.train()
            self.classifier.train()
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, "TRAIN", grad=True)

            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                # Set models to evaluation mode
                self.gcn.eval()
                self.classifier.eval()
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, "VALID", grad=False)
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    print("### w" + ") ep " + str(e) + " - Best valid measure:" + str(eval_valid))
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print("### w" + ") ep " + str(e) + " - Early stop.")
                        break

            if (
                len(self.splitter.test) > 0
                and eval_valid == best_eval_valid
                and e > self.args.eval_after_epochs
            ):
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
            pbar = tqdm.tqdm(split, desc=f"Epoch {epoch} - {set_name}", leave=True)
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
                print(set_name)
                if (
                    set_name.startswith("TEST") or set_name.startswith("VALID")
                ) and self.args.task == "link_pred":
                    self.logger.log_minibatch(
                        predictions, s.label_sp["vals"], loss.detach(), adj=s.label_sp["idx"]
                    )
                else:
                    self.logger.log_minibatch(predictions, s.label_sp["vals"], loss.detach())

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
        nodes_embs = self.gcn(hist_adj_list, hist_ndFeats_list, mask_list)

        # !
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
        sample = u.Namespace(sample)
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

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp["idx"] = label_sp["idx"].to(self.args.device).t()
        else:
            label_sp["idx"] = label_sp["idx"].to(self.args.device)

        label_sp["vals"] = label_sp["vals"].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def prepare_static_sample(self, sample):
        """Prepare a sample from static graph data.

        Args:
                sample: Raw sample from static graph.

        Returns:
                Processed sample.
        """
        sample = u.Namespace(sample)

        sample.hist_adj_list = self.hist_adj_list

        sample.hist_ndFeats_list = self.hist_ndFeats_list

        label_sp = {}
        label_sp["idx"] = [sample.idx]
        label_sp["vals"] = sample.label
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        """Remove batch dimension from adjacency dict.

        Args:
                adj: Adjacency dict with batch dimension.

        Returns:
                Adjacency dict without batch dimension.
        """
        if self.args.task in ["link_pred", "edge_cls"]:
            adj["idx"] = adj["idx"][0]
        adj["vals"] = adj["vals"][0]
        return adj

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
        train_test_pairs = self.splitter.get_train_test_pairs()
        num_phases = len(train_test_pairs)

        if num_phases < 1:
            raise ValueError("Incremental training requires at least 2 snapshots")

        print(f"\n{'=' * 60}")
        print(f"Starting Incremental Training with {num_phases + 1} snapshots")
        print(f"{'=' * 60}\n")

        all_results = []

        for phase_idx, (train_snapshot, test_snapshot) in enumerate(train_test_pairs):
            print(f"\n{'=' * 60}")
            if phase_idx == 0:
                print(f"Phase {phase_idx + 1}: Initial Training")
            else:
                print(f"Phase {phase_idx + 1}: Fine-tuning")
            print(f"Train on snapshot {phase_idx} -> Test on snapshot {phase_idx + 1}")
            print(f"{'=' * 60}\n")

            phase_results = self._run_incremental_phase(
                train_snapshot=train_snapshot,
                test_snapshot=test_snapshot,
                phase_idx=phase_idx,
                is_initial=(phase_idx == 0),
            )

            all_results.append(phase_results)

            # Save checkpoint after each phase
            log_dir = getattr(self.args, "log_dir", "log/")
            checkpoint_path = os.path.join(log_dir, f"checkpoint_phase_{phase_idx}.pth.tar")
            torch.save(
                {
                    "phase": phase_idx,
                    "gcn_dict": self.gcn.state_dict(),
                    "classifier_dict": self.classifier.state_dict(),
                    "gcn_optimizer": self.gcn_opt.state_dict(),
                    "classifier_optimizer": self.classifier_opt.state_dict(),
                    "results": phase_results,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            # Free memory: clear temporal features cache after each phase
            if hasattr(self.data, "clear_temporal_cache"):
                self.data.clear_temporal_cache()
                gc.collect()
                if self.args.use_cuda:
                    torch.cuda.empty_cache()
        # Print summary of all phases
        self._print_incremental_summary(all_results)

        return all_results

    def _run_incremental_phase(
        self,
        train_snapshot: torch.utils.data.DataLoader,
        test_snapshot: torch.utils.data.DataLoader,
        phase_idx: int,
        is_initial: bool,
    ) -> dict:
        """Run a single phase of incremental training.

        Args:
            train_snapshot: DataLoader for training data.
            test_snapshot: DataLoader for test data.
            phase_idx: Index of the current phase.
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
            "is_initial": is_initial,
            "num_epochs": num_epochs,
            "train_metrics": [],
            "test_metrics": [],
            "best_test_metric": 0,
            "best_epoch": 0,
        }

        for e in range(num_epochs):
            # Training
            self.gcn.train()
            self.classifier.train()
            eval_train, nodes_embs = self.run_epoch(
                train_snapshot, e, f"TRAIN (Phase {phase_idx})", grad=True
            )
            phase_results["train_metrics"].append(eval_train)

            # Testing
            self.gcn.eval()
            self.classifier.eval()
            eval_test, _ = self.run_epoch(
                test_snapshot, e, f"TEST (Phase {phase_idx})", grad=False
            )
            phase_results["test_metrics"].append(eval_test)

            print(f"Phase {phase_idx} - Epoch {e}: Train={eval_train:.4f}, Test={eval_test:.4f}")

            # Track best performance
            if eval_test > best_eval_test:
                best_eval_test = eval_test
                best_epoch = e
                epochs_without_impr = 0
                print(f"  -> New best test metric: {best_eval_test:.4f}")
            else:
                epochs_without_impr += 1

            # Early stopping
            if epochs_without_impr > self.args.early_stop_patience:
                print(f"  -> Early stopping at epoch {e}")
                break

        phase_results["best_test_metric"] = best_eval_test
        phase_results["best_epoch"] = best_epoch

        print(f"\nPhase {phase_idx} completed:")
        print(f"  Best test metric: {best_eval_test:.4f} at epoch {best_epoch}")

        return phase_results

    def _print_incremental_summary(self, all_results: list) -> None:
        """Print a summary of all incremental training phases.

        Args:
            all_results: List of results from each phase.
        """
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        for result in all_results:
            phase_type = "Initial Training" if result["is_initial"] else "Fine-tuning"
            print(f"\nPhase {result['phase_idx']} ({phase_type}):")
            print(f"  Best Test Metric: {result['best_test_metric']:.4f}")
            print(f"  Best Epoch: {result['best_epoch']}")
            print(f"  Total Epochs: {len(result['train_metrics'])}")

        # Calculate average performance across all test phases
        test_metrics = [r["best_test_metric"] for r in all_results]
        print(f"\n{'=' * 60}")
        print(f"Overall Statistics:")
        print(f"  Average Test Metric: {np.mean(test_metrics):.4f}")
        print(f"  Std Test Metric: {np.std(test_metrics):.4f}")
        print(f"  Min Test Metric: {np.min(test_metrics):.4f}")
        print(f"  Max Test Metric: {np.max(test_metrics):.4f}")
        print(f"{'=' * 60}\n")
