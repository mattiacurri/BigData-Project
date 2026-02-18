"""Logging utilities for training runs (console/file + optional wandb)."""

import datetime
import logging
import pprint
import sys
import time

from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from wandb_logger import WandbLogger

import utils


class Logger:
    """Logging helper that aggregates minibatch/epoch metrics and reports them."""

    def __init__(self, args, num_classes, minibatch_log_interval=10):
        """Configure logger output (file or stdout) and optional wandb integration."""
        if args is not None:
            currdate = str(datetime.datetime.today().strftime("%Y%m%d%H%M%S"))
            config_name = getattr(args, "config_file_name", "default")
            self.log_name = (
                "log/log_"
                + args.data
                + "_"
                + args.task
                + "_"
                + args.model
                + "_"
                + config_name
                + "_"
                + currdate
                + "_r"
                + ".log"
            )

            if args.use_logfile:
                print("Log file:", self.log_name)
                logging.basicConfig(filename=self.log_name, level=logging.INFO)
            else:
                print("Log: STDOUT")
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)

            logging.info("*** PARAMETERS ***")
            logging.info(pprint.pformat(args.__dict__))
            logging.info("")
        else:
            print("Log: STDOUT")
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.num_classes = num_classes
        self.minibatch_log_interval = minibatch_log_interval
        self.args = args

        self.wandb_logger = None
        if args is not None and getattr(args, "use_wandb", False):
            run_id = currdate if args is not None else "unknown"
            self.wandb_logger = WandbLogger(args, run_id)
            config_dict = args.__dict__.copy() if hasattr(args, "__dict__") else {}
            config_file_path = getattr(args, "config_file_path", None)
            self.wandb_logger.init(config_dict, config_file_path)

    def get_log_file_name(self):
        """Return current log filename."""
        return self.log_name

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        """Initialize epoch-level accumulators before processing minibatches."""
        # ALDO
        self.epoch = epoch
        ######
        self.set = set
        self.losses = []
        self.errors = []
        self.MAPs = []
        self.AUCs = []
        # self.time_step_sizes = []
        self.conf_mat_tp = {}
        self.conf_mat_fn = {}
        self.conf_mat_fp = {}
        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] = 0
            self.conf_mat_fn[cl] = 0
            self.conf_mat_fp[cl] = 0

        if self.set == "TEST":
            self.conf_mat_tp_list = {}
            self.conf_mat_fn_list = {}
            self.conf_mat_fp_list = {}
            for cl in range(self.num_classes):
                self.conf_mat_tp_list[cl] = []
                self.conf_mat_fn_list[cl] = []
                self.conf_mat_fp_list[cl] = []

        self.batch_sizes = []
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        logging.info("################ " + set + " epoch " + str(epoch) + " ###################")
        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

    def log_minibatch(self, predictions, true_classes, loss, **kwargs):
        """Process a minibatch: update metrics, confusion matrix and optional logging."""
        probs = torch.softmax(predictions, dim=1)[:, 1]

        filter_edges = kwargs.get("filter_edges")
        adj = kwargs.get("adj")

        if filter_edges is not None and adj is not None:
            # Vectorized filtering on GPU - avoid numpy conversion
            adj_edges = adj
            if adj_edges.shape[0] == 2 and adj_edges.shape[1] != 2:
                adj_edges = adj_edges.t()

            # Convert filter_edges to tensor hash for fast comparison
            filter_tensor = torch.tensor(
                filter_edges, dtype=adj_edges.dtype, device=adj_edges.device
            )
            if filter_tensor.dim() == 2 and filter_tensor.shape[1] == 2:
                # Use hash-based filtering: edge_id = source * large_constant + target
                large_const = 1000000
                adj_hashes = adj_edges[:, 0] * large_const + adj_edges[:, 1]
                filter_hashes = filter_tensor[:, 0] * large_const + filter_tensor[:, 1]
                mask = ~torch.isin(adj_hashes, filter_hashes)

            filtered_probs = probs[mask]
            filtered_true_classes = true_classes[mask]
            MAP = torch.tensor(
                self.get_MAP(filtered_probs, filtered_true_classes, do_softmax=False)
            )
        else:
            MAP = torch.tensor(self.get_MAP(probs, true_classes, do_softmax=False))

        AUC = torch.tensor(self.get_AUC(predictions, true_classes))

        error, conf_mat_per_class = self.eval_predicitions(
            predictions, true_classes, self.num_classes
        )

        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)

        if loss is not None:
            self.losses.append(loss)
        self.errors.append(error)
        self.MAPs.append(MAP)
        self.AUCs.append(AUC)
        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] += conf_mat_per_class.true_positives[cl]
            self.conf_mat_fn[cl] += conf_mat_per_class.false_negatives[cl]
            self.conf_mat_fp[cl] += conf_mat_per_class.false_positives[cl]
            if self.set == "TEST":
                self.conf_mat_tp_list[cl].append(conf_mat_per_class.true_positives[cl])
                self.conf_mat_fn_list[cl].append(conf_mat_per_class.false_negatives[cl])
                self.conf_mat_fp_list[cl].append(conf_mat_per_class.false_positives[cl])

        self.minibatch_done += 1
        if self.minibatch_done % self.minibatch_log_interval == 0:
            mb_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            if len(self.losses) > 0:
                partial_losses = torch.stack(self.losses)
                logging.info(
                    self.set
                    + " batch %d / %d - partial error %0.4f - partial loss %0.4f - partial MAP %0.4f"
                    % (
                        self.minibatch_done,
                        self.num_minibatches,
                        mb_error,
                        partial_losses.mean(),
                        mb_MAP,
                    )
                )
            else:
                logging.info(
                    self.set
                    + " batch %d / %d - partial error %0.4f - partial MAP %0.4f"
                    % (
                        self.minibatch_done,
                        self.num_minibatches,
                        mb_error,
                        mb_MAP,
                    )
                )

            tp = conf_mat_per_class.true_positives
            fn = conf_mat_per_class.false_negatives
            fp = conf_mat_per_class.false_positives
            logging.info(
                self.set
                + " batch %d / %d -  partial tp %s,fn %s,fp %s"
                % (self.minibatch_done, self.num_minibatches, tp, fn, fp)
            )
            precision, recall, f1 = self.calc_macroavg_eval_measures(tp, fn, fp, self.num_classes)
            logging.info(
                self.set
                + " batch %d / %d - measures partial macroavg - precision %0.4f - recall %0.4f - f1 %0.4f "
                % (self.minibatch_done, self.num_minibatches, precision, recall, f1)
            )
            for cl in range(self.num_classes):
                cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(tp, fn, fp, cl)
                logging.info(
                    self.set
                    + " batch %d / %d - measures partial for class %d - precision %0.4f - recall %0.4f - f1 %0.4f "
                    % (
                        self.minibatch_done,
                        self.num_minibatches,
                        cl,
                        cl_precision,
                        cl_recall,
                        cl_f1,
                    )
                )

            logging.info(
                self.set
                + " batch %d / %d - Batch time %d "
                % (
                    self.minibatch_done,
                    self.num_minibatches,
                    (time.monotonic() - self.lasttime),
                )
            )

        self.lasttime = time.monotonic()

    def log_epoch_done(self):
        """Finalize epoch metrics, log summaries and return the evaluation measure."""
        eval_measure = 0

        if len(self.losses) > 0:
            self.losses = torch.stack(self.losses)
            logging.info(self.set + " mean losses " + str(self.losses.mean()))
            if self.args.target_measure == "loss" or self.args.target_measure == "Loss":
                eval_measure = self.losses.mean()
        else:
            if self.args.target_measure == "loss" or self.args.target_measure == "Loss":
                eval_measure = 0

        epoch_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
        logging.info(self.set + " mean errors " + str(epoch_error))

        epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        logging.info(self.set + " - mean MAP " + str(epoch_MAP))

        if self.args.target_measure == "MAP" or self.args.target_measure == "map":
            eval_measure = epoch_MAP

        epoch_AUC = self.calc_epoch_metric(self.batch_sizes, self.AUCs)
        logging.info(self.set + " - mean AUC " + str(epoch_AUC))
        if self.args.target_measure == "AUC" or self.args.target_measure == "auc":
            eval_measure = epoch_AUC

        logging.info(
            self.set
            + " tp %s,fn %s,fp %s" % (self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp)
        )
        precision, recall, f1 = self.calc_macroavg_eval_measures(
            self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, self.num_classes
        )
        logging.info(
            self.set
            + " measures macroavg - precision %0.4f - recall %0.4f - f1 %0.4f "
            % (precision, recall, f1)
        )
        if self.args.target_measure in ["macro_f1", "Macro_F1", "MACRO_F1"]:
            eval_measure = f1
        elif str(self.args.target_class) == "AVG":
            if self.args.target_measure in ["Precision", "prec"]:
                eval_measure = precision
            elif self.args.target_measure in ["Recall", "rec"]:
                eval_measure = recall
            elif self.args.target_measure in ["f1", "F1"]:
                eval_measure = f1

        for cl in range(self.num_classes):
            cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(
                self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, cl
            )
            logging.info(
                self.set
                + " measures for class %d - precision %0.4f - recall %0.4f - f1 %0.4f "
                % (cl, cl_precision, cl_recall, cl_f1)
            )
            if str(cl) == str(self.args.target_class):
                if self.args.target_measure in [
                    "MAP",
                    "map",
                    "AUC",
                    "auc",
                    "Loss",
                    "loss",
                    "macro_f1",
                    "Macro_F1",
                    "MACRO_F1",
                ]:
                    pass
                elif self.args.target_measure in ["Precision", "prec"]:
                    eval_measure = cl_precision
                elif self.args.target_measure in ["Recall", "rec"]:
                    eval_measure = cl_recall
                elif self.args.target_measure in ["f1", "F1"]:
                    eval_measure = cl_f1

        logging.info(self.set + " Total epoch time: " + str((time.monotonic() - self.ep_time)))

        if self.wandb_logger is not None:
            class_metrics = {}
            for cl in range(self.num_classes):
                cl_p, cl_r, cl_f1 = self.calc_eval_measures_per_class(
                    self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, cl
                )
                class_metrics[f"class_{cl}/precision"] = cl_p
                class_metrics[f"class_{cl}/recall"] = cl_r
                class_metrics[f"class_{cl}/f1"] = cl_f1
                class_metrics[f"class_{cl}/tp"] = self.conf_mat_tp[cl].item()
                class_metrics[f"class_{cl}/fn"] = self.conf_mat_fn[cl].item()
                class_metrics[f"class_{cl}/fp"] = self.conf_mat_fp[cl].item()

            total_tp = sum(self.conf_mat_tp[cl].item() for cl in range(self.num_classes))
            total_fn = sum(self.conf_mat_fn[cl].item() for cl in range(self.num_classes))
            total_fp = sum(self.conf_mat_fp[cl].item() for cl in range(self.num_classes))
            total_samples = total_tp + total_fn + total_fp
            accuracy = total_tp / total_samples if total_samples > 0 else 0.0

            metrics = {
                "error": epoch_error,
                "mean_error": epoch_error,
                "MAP": epoch_MAP,
                "AUC": epoch_AUC,
                "accuracy": accuracy,
                "tp": total_tp,
                "fn": total_fn,
                "fp": total_fp,
                "macro/precision": precision,
                "macro/recall": recall,
                "macro/f1": f1,
                "epoch_time": time.monotonic() - self.ep_time,
            }
            if len(self.losses) > 0:
                metrics["loss"] = self.losses.mean().item()
                metrics["mean_loss"] = self.losses.mean().item()
            metrics.update(class_metrics)

            self.wandb_logger.log_epoch_metrics(self.set, self.epoch, metrics)

        return eval_measure

    def get_MAP(self, predictions, true_classes, do_softmax=False):
        """Return mean average precision (MAP) for `predictions` vs `true_classes`."""
        probs = torch.softmax(predictions, dim=1)[:, 1] if do_softmax else predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        if len(true_classes_np) == 0:
            return 0.0

        return average_precision_score(true_classes_np, predictions_np)

    def get_AUC(self, predictions, true_classes):
        """Return AUC for `predictions` vs `true_classes` (binary classification)."""
        probs = torch.sigmoid(predictions)[
            :, 1
        ]  # Assuming binary classification, take positive class prob
        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()
        if len(true_classes_np) == 0 or len(predictions_np) == 0:
            return 0.0
        return roc_auc_score(true_classes_np, predictions_np)

    def eval_predicitions(self, predictions, true_classes, num_classes):
        """Compute error and per-class confusion matrix entries for `predictions`."""
        if predictions.size(0) == 0:
            error = 0.0
            predicted_classes = torch.empty(0, dtype=torch.long, device=predictions.device)
        else:
            predicted_classes = predictions.argmax(dim=1)
            failures = (predicted_classes != true_classes).sum(dtype=torch.float)
            error = failures / predictions.size(0)

        error = error.detach().clone() if isinstance(error, torch.Tensor) else torch.tensor(error)

        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        for cl in range(num_classes):
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = predicted_classes[cl_indices] == true_classes[cl_indices]

            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return error, conf_mat_per_class

    def eval_predicitions_at_k(self, predictions, true_classes, num_classes, k):
        """Compute top-k confusion metrics per class for recommendation-style evaluation."""
        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        if predictions.size(0) < k:
            k = predictions.size(0)

        for cl in range(num_classes):
            # sort for prediction with higher score for target class (cl)
            _, idx_preds_at_k = torch.topk(predictions[:, cl], k, dim=0, largest=True, sorted=True)
            predictions_at_k = predictions[idx_preds_at_k]
            predicted_classes = predictions_at_k.argmax(dim=1)

            cl_indices_at_k = true_classes[idx_preds_at_k] == cl
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (
                predicted_classes[cl_indices_at_k] == true_classes[idx_preds_at_k][cl_indices_at_k]
            )

            tp = hits.sum()
            fn = (
                true_classes[cl_indices].size(0) - tp
            )  # This only if we want to consider the size at K -> hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return conf_mat_per_class

    def calc_microavg_eval_measures(self, tp, fn, fp):
        """Compute micro-averaged precision, recall and F1 from tp/fn/fp maps."""
        tp_sum = sum(tp.values()).item()
        fn_sum = sum(fn.values()).item()
        fp_sum = sum(fp.values()).item()

        p = 0.0 if tp_sum + fp_sum == 0 else tp_sum * 1.0 / (tp_sum + fp_sum)
        r = 0.0 if tp_sum + fn_sum == 0 else tp_sum * 1.0 / (tp_sum + fn_sum)
        f1 = 2.0 * (p * r) / (p + r) if p + r > 0 else 0
        return p, r, f1

    def calc_eval_measures_per_class(self, tp, fn, fp, class_id):
        """Compute precision, recall and F1 for a single `class_id` from tp/fn/fp."""
        # ALDO
        if type(tp) is dict:
            tp_sum = tp[class_id].item()
            fn_sum = fn[class_id].item()
            fp_sum = fp[class_id].item()
        else:
            tp_sum = tp.item()
            fn_sum = fn.item()
            fp_sum = fp.item()
        ########
        if tp_sum == 0:
            return 0, 0, 0

        p = tp_sum * 1.0 / (tp_sum + fp_sum)
        r = tp_sum * 1.0 / (tp_sum + fn_sum)
        f1 = 2.0 * (p * r) / (p + r) if p + r > 0 else 0
        return p, r, f1

    def calc_macroavg_eval_measures(self, tp, fn, fp, num_classes):
        """Compute macro-averaged precision, recall and F1 across classes."""
        precisions = []
        recalls = []
        f1s = []
        for cl in range(num_classes):
            p, r, f1 = self.calc_eval_measures_per_class(tp, fn, fp, cl)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        p_macro = sum(precisions) / len(precisions) if precisions else 0
        r_macro = sum(recalls) / len(recalls) if recalls else 0
        f1_macro = sum(f1s) / len(f1s) if f1s else 0
        return p_macro, r_macro, f1_macro

    def calc_epoch_metric(self, batch_sizes, metric_val):
        """Compute weighted average of minibatch metric values across an epoch."""
        batch_sizes = torch.tensor(batch_sizes, dtype=torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum() / batch_sizes.sum()

        return epoch_metric_val.detach().item()
