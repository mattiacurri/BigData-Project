"""Trainer and training loop utilities for EvolveGCN experiments."""

import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import logger
import utils as u


class Trainer:
    """Training loop, checkpointing and dataset helpers for EvolveGCN."""

    def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        """Initialize Trainer.

        Args:
            args: runtime arguments namespace.
            splitter: data splitter providing train/dev/test splits.
            gcn: graph convolutional model instance.
            classifier: classifier head.
            comp_loss: loss function.
            dataset: dataset object.
            num_classes: number of target classes.
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

        # Create unique save directory for this run
        log_file_name = self.logger.get_log_file_name()
        self.run_id = log_file_name.split("log_")[1].split(".log")[0]
        self.save_dir = f"models_and_embeddings/{self.run_id}/"
        os.makedirs(self.save_dir, exist_ok=True)

        self.init_optimizers(args)

        self.tr_step = 0

    def init_optimizers(self, args):
        """Create optimizers for GCN and classifier using settings from `args`."""
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-4)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-4)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def save_checkpoint(self, state, filename="checkpoint.pth.tar"):
        """Save `state` dict to `self.save_dir` under `filename`."""
        torch.save(state, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename, model):
        """Load checkpoint from `self.save_dir`; return `epoch` when found, otherwise 0."""
        full_path = os.path.join(self.save_dir, filename)
        if os.path.isfile(full_path):
            print("=> loading checkpoint '{}'".format(full_path))
            checkpoint = torch.load(full_path)
            epoch = checkpoint["epoch"]
            if "gcn_dict" in checkpoint:
                self.gcn.load_state_dict(checkpoint["gcn_dict"])
                self.classifier.load_state_dict(checkpoint["classifier_dict"])
            else:
                # Load whole model
                self.gcn = checkpoint["gcn"]
                self.classifier = checkpoint["classifier"]
                self.gcn.to(self.args.device)
                self.classifier.to(self.args.device)

            # Reset random seeds after loading to ensure deterministic behavior
            torch.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.args.seed)
                torch.cuda.manual_seed_all(self.args.seed)

            print("=> loaded checkpoint '{}' (epoch {})".format(full_path, checkpoint["epoch"]))
            return epoch
        print("=> no checkpoint found at '{}'".format(full_path))
        return 0

    def train(self):
        """Run the main training loop, track best model and optionally evaluate on test set."""
        best_eval_valid = 0
        best_epoch = -1
        epochs_without_impr = 0
        valid_measures = []

        for e in range(self.args.num_epochs):
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, "TRAIN", grad=True)

            if len(self.splitter.dev) > 0:
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, "VALID", grad=False)
                valid_measures.append((e, eval_valid))
                print(f"MAP after VALID epoch {e}: {eval_valid}")
                print(f"eval_valid: {eval_valid}, best_Eval_Valid: {best_eval_valid}")
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    best_epoch = e
                    epochs_without_impr = 0
                    print("### w" + ") ep " + str(e) + " - Best valid measure:" + str(eval_valid))
                    # Save best model
                    try:
                        self.save_checkpoint(
                            {
                                "epoch": e,
                                "gcn_dict": self.gcn.state_dict(),
                                "classifier_dict": self.classifier.state_dict(),
                            },
                            f"best_model_{e}.pth.tar",
                        )
                    except Exception as ex:
                        print(f"Failed to save checkpoint: {ex}")
                        # Save manually
                        torch.save(
                            {
                                "epoch": e,
                                "gcn": self.gcn,
                                "classifier": self.classifier,
                            },
                            os.path.join(self.save_dir, f"best_model_{e}.pth.tar"),
                        )
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print("### w" + ") ep " + str(e) + " - Early stop.")
                        break

            if len(self.splitter.test) > 0:
                # Skip TEST during training for LOOCV (will be computed at end with best model)
                split_mode = getattr(self.args, "split_mode", "proportion")
                if split_mode != "loocv":
                    eval_test, _ = self.run_epoch(self.splitter.test, e, "TEST", grad=False)
                    print(f"MAP after TEST epoch {e}: {eval_test}")

                # if self.args.save_node_embeddings:
                # 	self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, f"{log_file}_train_nodeembs_{e}.csv.gz")
                # 	self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, f"{log_file}_valid_nodeembs_{e}.csv.gz")
                # 	self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, f"{log_file}_test_nodeembs_{e}.csv.gz")

        if valid_measures:
            best_epoch = max(valid_measures, key=lambda x: x[1])[0]
        logging.info("Best epoch: " + str(best_epoch + 1))

        best_test_map = None
        # Load best model and run test
        if os.path.exists(os.path.join(self.save_dir, f"best_model_{best_epoch}.pth.tar")):
            self.load_checkpoint(f"best_model_{best_epoch}.pth.tar", None)
            # Set models to eval mode after loading
            self.gcn.eval()
            self.classifier.eval()
            # resave it by renaming as best model best epoch best
            try:
                self.save_checkpoint(
                    {
                        "epoch": best_epoch,
                        "gcn_dict": self.gcn.state_dict(),
                        "classifier_dict": self.classifier.state_dict(),
                    },
                    "best_model.pth.tar",
                )
            except Exception as ex:
                print(f"Failed to rename checkpoint: {ex}")

            print(f"Loaded best model from epoch {best_epoch + 1}")
            if self.args.save_node_embeddings:
                _, nodes_embs = self.run_epoch(
                    self.splitter.train, best_epoch, "TRAIN_BEST", grad=False
                )
                self.save_node_embs_csv(
                    nodes_embs,
                    self.splitter.train_idx,
                    os.path.join(self.save_dir, "best_train_nodeembs.csv.gz"),
                )
                self.save_node_embs_csv(
                    nodes_embs,
                    self.splitter.dev_idx,
                    os.path.join(self.save_dir, "best_valid_nodeembs.csv.gz"),
                )
                self.save_node_embs_csv(
                    nodes_embs,
                    self.splitter.test_idx,
                    os.path.join(self.save_dir, "best_test_nodeembs.csv.gz"),
                )
            if len(self.splitter.test) > 0:
                eval_test, _ = self.run_epoch(
                    self.splitter.test, best_epoch, "TEST_BEST", grad=False
                )
                best_test_map = eval_test
                print(f"Test MAP with best model: {eval_test}")
                self.save_test_predictions()

        self._log_wandb_artifacts(best_epoch, best_test_map)

    def _log_wandb_artifacts(self, best_epoch, best_test_map=None):
        if self.logger.wandb_logger is None:
            return

        wandb_logger = self.logger.wandb_logger

        best_model_path = os.path.join(self.save_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            wandb_logger.log_artifact(
                best_model_path,
                artifact_type="model",
                name=f"best_model_{self.run_id}",
                description=f"Best model from epoch {best_epoch + 1}",
            )

        for split in ["train", "valid", "test"]:
            emb_path = os.path.join(self.save_dir, f"best_{split}_nodeembs.csv.gz")
            if os.path.exists(emb_path):
                wandb_logger.log_artifact(
                    emb_path,
                    artifact_type="embeddings",
                    name=f"{split}_embeddings_{self.run_id}",
                    description=f"Node embeddings for {split} set",
                )

        predictions_path = os.path.join(self.save_dir, "test_predictions.csv")
        if os.path.exists(predictions_path):
            wandb_logger.save_file(predictions_path)

        if self.args.config_file_path and os.path.exists(self.args.config_file_path):
            wandb_logger.save_file(self.args.config_file_path)

        wandb_logger.log_summary("best_epoch", best_epoch + 1)
        if best_test_map is not None:
            wandb_logger.log_summary("best_test_MAP", best_test_map)
        wandb_logger.finish()

    def run_epoch(self, split, epoch, set_name, grad):
        """Execute one epoch on `split`. Returns (eval_measure, nodes_embs)."""
        log_interval = 999
        if set_name == "TEST":
            log_interval = 1
        self.logger.log_epoch_start(
            epoch, len(split), set_name, minibatch_log_interval=log_interval
        )

        torch.set_grad_enabled(grad)
        # Set model mode: train() for training, eval() for validation/test
        if grad:
            self.gcn.train()
            self.classifier.train()
        else:
            self.gcn.eval()
            self.classifier.eval()

        self.epoch_predictions = []
        pbar = tqdm(split, desc=f"{set_name} Epoch {epoch}", total=len(split))
        for s in pbar:
            s = self.prepare_sample(s)

            predictions, nodes_embs = self.predict(
                s.hist_adj_list,
                s.hist_ndFeats_list,
                s.label_sp["idx"],
                s.node_mask_list,
            )

            if grad:
                loss = self.comp_loss(predictions, s.label_sp["vals"])
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                loss = None
                pbar.set_postfix()

            if set_name in ["TEST", "VALID", "TEST_BEST"] and self.args.task == "link_pred":
                filter_edges = getattr(s, "filter_edges", None)
                self.logger.log_minibatch(
                    predictions,
                    s.label_sp["vals"],
                    loss,
                    adj=s.label_sp["idx"],
                    filter_edges=filter_edges,
                )
            else:
                self.logger.log_minibatch(predictions, s.label_sp["vals"], loss)
            if grad:
                self.optim_step(loss)

            if "TEST" in set_name:
                self.epoch_predictions.append(
                    (predictions.detach(), s.label_sp["idx"], s.label_sp["vals"])
                )

        pbar.close()
        torch.set_grad_enabled(True)
        eval_measure = self.logger.log_epoch_done()

        if "TEST" in set_name:
            self.test_predictions = self.epoch_predictions

        return eval_measure, nodes_embs

    def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, mask_list):
        """Compute node embeddings using GCN and return classifier predictions for `node_indices`."""
        nodes_embs = self.gcn(hist_adj_list, hist_ndFeats_list, mask_list)

        # Debug: Check for existing edges in historical adj (COMMENTED FOR PERFORMANCE)
        # existing_edges = set()
        # for adj in hist_adj_list:
        #     edges = adj.coalesce().indices().t().cpu().numpy()
        #     for e in edges:
        #         existing_edges.add((e[0], e[1]))
        # predict_edges = node_indices.t().cpu().numpy()
        # seen_count = 0
        # first_seen = None
        # self_follow_count = 0
        # for e in predict_edges:
        #     if (e[0], e[1]) in existing_edges:
        #         seen_count += 1
        #         if first_seen is None:
        #             first_seen = e
        #         if e[0] == e[1]:
        #             self_follow_count += 1
        # if seen_count > 0:
        #     print(f" DEBUG: Predicting on {len(predict_edges)} edges, {seen_count} already seen")

        predict_batch_size = 100000
        gather_predictions = []
        for i in range(1 + (node_indices.size(1) // predict_batch_size)):
            batch_indices = node_indices[:, i * predict_batch_size : (i + 1) * predict_batch_size]
            cls_input = torch.cat(
                [nodes_embs[batch_indices[0]], nodes_embs[batch_indices[1]]], dim=1
            )
            predictions = self.classifier(cls_input)
            gather_predictions.append(predictions)
        gather_predictions = torch.cat(gather_predictions, dim=0)

        # Detach if not in training mode to save memory
        if not torch.is_grad_enabled():
            gather_predictions = gather_predictions.detach()
            nodes_embs = nodes_embs.detach()
        return gather_predictions, nodes_embs

    # def gather_node_embs(self,nodes_embs,node_indices):
    # 	emb1 = nodes_embs[node_indices[0]]
    # 	emb2 = nodes_embs[node_indices[1]]
    # 	return torch.cat([emb1, emb2], dim=1)

    def optim_step(self, loss):
        """Backward on `loss` and perform optimizer step according to accumulation schedule."""
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

    def prepare_sample(self, sample):
        """Move sample tensors to `self.args.device` and prepare node features/masks."""
        sample = u.Namespace(sample)
        for i, adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])
            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            sample.node_mask_list[i] = node_mask.to(
                self.args.device
            ).t()  # transposed to have same dimensions as scorer

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp["idx"] = (
                label_sp["idx"].to(self.args.device).t()
            )  ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
        else:
            label_sp["idx"] = label_sp["idx"].to(self.args.device)

        label_sp["vals"] = label_sp["vals"].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp
        return sample

    def prepare_static_sample(self, sample):
        """Prepare a static single sample using stored `hist_adj_list` and features."""
        sample = u.Namespace(sample)

        sample.hist_adj_list = self.hist_adj_list

        sample.hist_ndFeats_list = self.hist_ndFeats_list

        label_sp = {}
        label_sp["idx"] = [sample.idx]
        label_sp["vals"] = sample.label
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        """Remove batch dimension from sparse-adj-like dict `adj` in-place."""
        if self.args.task in ["link_pred", "edge_cls"]:
            adj["idx"] = adj["idx"][0]
        adj["vals"] = adj["vals"][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        """Save node embeddings (with original IDs) to a gzipped CSV at `file_name`."""
        print(f"Saving node embeddings to {file_name} for {len(indexes)} nodes.")
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]]).to(
                nodes_embs.device
            )
            emb = torch.cat((orig_ID, nodes_embs[node_id].double())).detach().cpu().numpy()
            csv_node_embs.append(emb)
        pd.DataFrame(np.array(csv_node_embs)).to_csv(
            file_name, header=None, index=None, compression="gzip"
        )

    def save_test_predictions(self):
        """Persist accumulated test predictions to CSV and append summary counts."""
        if hasattr(self, "test_predictions") and self.test_predictions:
            all_preds = torch.cat([p for p, _, _ in self.test_predictions])
            all_edges = torch.cat([e for _, e, _ in self.test_predictions], dim=1)
            all_labels = torch.cat([labels for _, _, labels in self.test_predictions])
            probs = torch.sigmoid(all_preds)
            predictions_binary = (probs[:, 1] >= 0.5).long()
            num_pos = (predictions_binary == 1).sum().item()
            num_neg = (predictions_binary == 0).sum().item()
            df = pd.DataFrame(
                {
                    "source": all_edges[0].cpu().numpy(),
                    "target": all_edges[1].cpu().numpy(),
                    "prob": probs[:, 1].cpu().numpy(),
                    "ground_truth": all_labels.cpu().numpy(),
                    "prediction": predictions_binary.cpu().numpy(),
                }
            )
            # Aggiungi righe per num_pos e num_neg
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "source": ["NUM_POS", "NUM_NEG"],
                            "target": [num_pos, num_neg],
                            "prob": [None, None],
                            "ground_truth": [None, None],
                            "prediction": [None, None],
                        }
                    ),
                ],
                ignore_index=True,
            )
            df.to_csv(os.path.join(self.save_dir, "test_predictions.csv"), index=False)
            print(
                f"Test predictions saved to {os.path.join(self.save_dir, 'test_predictions.csv')}"
            )
        else:
            print("No test_predictions to save")
