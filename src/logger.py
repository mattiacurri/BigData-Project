"""Logger for training and evaluation metrics.

Provides comprehensive logging of training progress, evaluation metrics including
precision, recall, F1-score, MAP, and AUC-ROC across different datasets and phases.
Features: colored console output, timestamps, ASCII tables.
"""

import datetime
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Dict

from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import wandb


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
        "RESET": "\033[0m",
        # Additional colors for metrics
        "HEADER": "\033[1;35m",  # Bold Magenta
        "METRIC": "\033[36m",  # Cyan
        "SUCCESS": "\033[1;32m",  # Bold Green
    }

    def format(self, record):
        """Format log record with ANSI color codes based on log level.

        Args:
            record: LogRecord object to format.

        Returns:
            str: Formatted log message with color codes.
        """
        # Get color based on level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format the message
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)."""


def format_table(headers, rows, title=None):
    """Create an ASCII table with the given headers and rows.

    Args:
        headers: List of column headers.
        rows: List of rows, each row is a list of values.
        title: Optional title for the table.

    Returns:
        str: Formatted ASCII table.
    """
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding
    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths) + len(headers) + 1

    lines = []

    # Top border
    lines.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")

    # Title if provided
    if title:
        title_padded = f" {title} ".center(total_width - 2)
        lines.append(f"│{title_padded}│")
        lines.append("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Header row
    header_cells = [f" {str(h).center(w - 2)} " for h, w in zip(headers, col_widths)]
    lines.append("│" + "│".join(header_cells) + "│")
    lines.append("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Data rows
    for row in rows:
        cells = [f" {str(cell).center(w - 2)} " for cell, w in zip(row, col_widths)]
        lines.append("│" + "│".join(cells) + "│")

    # Bottom border
    lines.append("└" + "┴".join("─" * w for w in col_widths) + "┘")

    return "\n".join(lines)


def format_metrics_table(set_name, epoch, metrics_dict):
    """Format epoch metrics as an ASCII table.

    Args:
        set_name: Dataset split name (TRAIN/VALID/TEST).
        epoch: Epoch number.
        metrics_dict: Dictionary of metric names to values.

    Returns:
        str: Formatted metrics table.
    """
    headers = list(metrics_dict.keys())
    values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics_dict.values()]

    return format_table(headers, [values], title=f"{set_name} Epoch {epoch}")


class Logger:
    """Logger for experiment metrics and training progress."""

    def __init__(self, args, num_classes=2, minibatch_log_interval=10):
        """Initialize the logger.

        Args:
            args: Configuration namespace with logging parameters.
            num_classes: Number of classes for classification tasks.
            minibatch_log_interval: Interval (in batches) for logging during training.
        """
        if args is not None:
            currdate = str(datetime.datetime.today().strftime("%Y%m%d%H%M%S"))
            self.log_name = (
                "log/log_" + args.data + "_" + args.model + "_" + currdate + "_r" + ".log"
            )

            # Setup logging to BOTH stdout and file
            os.makedirs("log", exist_ok=True)

            # Get the root logger and clear any existing handlers
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.handlers.clear()

            # File handler (plain text with timestamp)
            file_formatter = PlainFormatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler = logging.FileHandler(self.log_name, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Stdout handler (colored with timestamp)
            console_formatter = ColoredFormatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.INFO)
            stdout_handler.setFormatter(console_formatter)
            stdout_handler.setFormatter(console_formatter)
            root_logger.addHandler(stdout_handler)
            self.stdout_handler = stdout_handler

        else:
            self.log_name = None
            print("Log: STDOUT only")
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.num_classes = num_classes
        self.minibatch_log_interval = minibatch_log_interval
        self.eval_k_list = [10, 100, 1000]
        self.args = args
        self.global_step = 0  # Global step counter for WandB
        self.phase_idx = 0
        self.phase_desc = "initial"

        # Initialize WandB if not already initialized
        if args is not None and not wandb.run:
            try:
                wandb.init(
                    project=args.data,
                    name=f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=vars(args),
                    reinit=False,
                    save_code=False,
                )
            except Exception as e:
                logging.warning(f"Failed to initialize WandB: {e}")

    def set_phase(self, phase_idx, phase_desc):
        """Set current phase for incremental logging.

        Args:
            phase_idx: Phase index number.
            phase_desc: Phase description string.
        """
        self.phase_idx = phase_idx
        self.phase_desc = phase_desc
        if wandb.run:
            try:
                # Log phase info as summary (can change over time) instead of config
                wandb.summary["phase_idx"] = phase_idx
                wandb.summary["phase_desc"] = phase_desc
            except Exception as e:
                logging.warning(f"Failed to log phase to WandB: {e}")

    def log_str(self, message, level="info"):
        """Log a string message with specified level.

        Args:
            message: Message to log.
            level: Log level (debug, info, warning, error).
        """
        level_func = getattr(logging, level.lower(), logging.info)
        level_func(message)

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        """Initialize logging for a new epoch.

        Args:
            epoch: Epoch number.
            num_minibatches: Total number of minibatches in the epoch.
            set: Dataset split (TRAIN/VALID/TEST).
            num_minibatches: Total number of minibatches in the epoch.
            set: Dataset split (TRAIN/VALID/TEST).
            minibatch_log_interval: Optional override for logging interval.
            console_log: Whether to log to console this epoch.
        """
        self.epoch = epoch
        self.set = set
        self.console_log = True
        self.verbose = True

        is_train = self.set.startswith("TRAIN")

        # Override verbose for validation/test
        if not is_train:
            self.verbose = True

        self.losses = []
        self.errors = []
        self.MAPs = []
        self.AUCs = []  # AUC-ROC on sampled negatives

        # Track number of samples for MAP
        self.MAP_sample_counts = []

        self.conf_mat_tp = {}
        self.conf_mat_fn = {}
        self.conf_mat_fp = {}
        self.conf_mat_tp_at_k = {}
        self.conf_mat_fn_at_k = {}
        self.conf_mat_fp_at_k = {}

        for k in self.eval_k_list:
            self.conf_mat_tp_at_k[k] = {}
            self.conf_mat_fn_at_k[k] = {}
            self.conf_mat_fp_at_k[k] = {}

        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] = 0
            self.conf_mat_fn[cl] = 0
            self.conf_mat_fp[cl] = 0
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl] = 0
                self.conf_mat_fn_at_k[k][cl] = 0
                self.conf_mat_fp_at_k[k][cl] = 0

        if self.set.startswith("TEST"):
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

        if self.verbose:
            logging.info(f"{set} EPOCH {epoch}")

        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

        # Initialize buffer for graph predictions on TEST set
        if self.set.startswith("TEST"):
            self.pred_edges_idx = []  # List of [2, batch_size] tensors
            self.pred_edges_vals = []  # List of [batch_size] tensors (binary predictions)
            self.pred_edge_probs = []  # List of [batch_size] tensors (probabilities)
            self.pred_edges_labels = []  # List of [batch_size] tensors (ground truth labels)

    def log_minibatch(self, predictions, true_classes, loss, **kwargs):
        """Log metrics for a single minibatch.

        Args:
            predictions: Model predictions (logits or probabilities).
            true_classes: Ground truth labels.
            loss: Loss value for the batch.
            **kwargs: Additional arguments (e.g., adj for link prediction).
        """
        probs = torch.softmax(predictions, dim=1)[:, 1]

        MAP = torch.tensor(self.get_MAP(probs, true_classes, do_softmax=False))
        # MAP is calculated on all samples
        self.last_map_num_samples = len(true_classes)

        # Calculate AUC-ROC on the sampled edges (positive + negative)
        # This is "sampling-based AUC" - measures discrimination on evaluated pairs only
        AUC = torch.tensor(self.get_AUC_ROC(probs, true_classes, do_softmax=False))

        error, conf_mat_per_class = self.eval_predicitions(
            predictions, true_classes, self.num_classes
        )
        conf_mat_per_class_at_k = {}
        for k in self.eval_k_list:
            conf_mat_per_class_at_k[k] = self.eval_predicitions_at_k(
                predictions, true_classes, self.num_classes, k
            )

        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)

        self.losses.append(loss)
        self.errors.append(error)
        self.MAPs.append(MAP)
        self.AUCs.append(AUC)

        # Track sample counts for MAP
        self.MAP_sample_counts.append(self.last_map_num_samples)

        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] += conf_mat_per_class.true_positives[cl]
            self.conf_mat_fn[cl] += conf_mat_per_class.false_negatives[cl]
            self.conf_mat_fp[cl] += conf_mat_per_class.false_positives[cl]
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl] += conf_mat_per_class_at_k[k].true_positives[cl]
                self.conf_mat_fn_at_k[k][cl] += conf_mat_per_class_at_k[k].false_negatives[cl]
                self.conf_mat_fp_at_k[k][cl] += conf_mat_per_class_at_k[k].false_positives[cl]
            if self.set.startswith("TEST"):
                self.conf_mat_tp_list[cl].append(conf_mat_per_class.true_positives[cl])
                self.conf_mat_fn_list[cl].append(conf_mat_per_class.false_negatives[cl])
                self.conf_mat_fp_list[cl].append(conf_mat_per_class.false_positives[cl])

        self.minibatch_done += 1
        self.global_step += 1

        if self.minibatch_done % self.minibatch_log_interval == 0:
            mb_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            partial_losses = torch.stack(self.losses)
            mb_loss = partial_losses.mean().item()

            if self.verbose:
                logging.info(
                    f"  📦 Batch {self.minibatch_done}/{self.num_minibatches} | "
                    f"Loss: {mb_loss:.4f} | "
                    f"Err: {mb_error:.4f} | "
                    f"MAP: {mb_MAP:.4f}"
                )

            # Log minibatch metrics to WandB
            if wandb.run:
                try:
                    prefix = f"{self.set.lower()}/minibatch/"
                    wandb.log(
                        {
                            f"{prefix}loss": mb_loss,
                            f"{prefix}error": mb_error,
                            f"{prefix}map": mb_MAP,
                            f"{prefix}batch": self.minibatch_done,
                        },
                        step=self.global_step,
                    )
                except Exception as e:
                    logging.warning(f"Failed to log minibatch to WandB: {e}")

        # Accumulate predictions for graph metrics calculation on TEST set
        if self.set.startswith("TEST"):
            probs = torch.softmax(predictions, dim=1)[:, 1]
            pred_binary = (probs > 0.5).long()

            if "adj" in kwargs:
                self.pred_edges_idx.append(kwargs["adj"].cpu())
                self.pred_edges_vals.append(pred_binary.cpu())
                self.pred_edge_probs.append(probs.cpu())
                self.pred_edges_labels.append(true_classes.cpu())

        self.lasttime = time.monotonic()

    def log_epoch_done(self):
        """Finalize logging for the completed epoch.

        Returns:
            float: Primary evaluation metric for the epoch.
        """
        eval_measure = 0

        self.losses = torch.stack(self.losses)
        mean_loss = float(self.losses.mean())

        # Calculate all metrics
        epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        epoch_AUC = self.calc_epoch_metric(self.batch_sizes, self.AUCs)

        # Calculate total number of samples used for MAP
        total_map_samples = sum(self.MAP_sample_counts)

        # Calculate per-class metrics
        class_metrics = {}
        for cl in range(self.num_classes):
            cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(
                self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, cl
            )
            class_metrics[cl] = {"precision": cl_precision, "recall": cl_recall, "f1": cl_f1}

        # Calculate macro-averaged metrics
        macro_precision = sum(m["precision"] for m in class_metrics.values()) / self.num_classes
        macro_recall = sum(m["recall"] for m in class_metrics.values()) / self.num_classes
        macro_f1 = sum(m["f1"] for m in class_metrics.values()) / self.num_classes

        # Simple logic: choose metric based on target_measure
        target = self.args.target_measure.lower()

        if target == "loss":
            eval_measure = mean_loss
        elif target == "map":
            eval_measure = epoch_MAP
        elif target in ["auc", "auc_roc"]:
            eval_measure = epoch_AUC
        elif target in ["precision", "prec"]:
            if str(self.args.target_class) == "AVG":
                eval_measure = macro_precision
            else:
                eval_measure = class_metrics[int(self.args.target_class)]["precision"]
        elif target in ["recall", "rec"]:
            if str(self.args.target_class) == "AVG":
                eval_measure = macro_recall
            else:
                eval_measure = class_metrics[int(self.args.target_class)]["recall"]
        elif target == "f1":
            if str(self.args.target_class) == "AVG":
                eval_measure = macro_f1
            else:
                eval_measure = class_metrics[int(self.args.target_class)]["f1"]
        else:
            raise ValueError(
                f"Unknown target_measure: '{self.args.target_measure}'. "
                f"Must be one of: loss, map, precision/prec, recall/rec, f1"
            )

        # Use macro-averaged for overall metrics
        precision, recall, f1 = macro_precision, macro_recall, macro_f1

        # ASCII Table Summary
        main_metrics = {
            "Loss": f"{mean_loss:.4f}",
            "Macro Prec": f"{precision:.4f}",
            "Macro Rec": f"{recall:.4f}",
            "Macro F1": f"{f1:.4f}",
            "MAP": f"{epoch_MAP:.4f}",
            "AUC-ROC": f"{epoch_AUC:.4f}",
        }

        table = format_metrics_table(self.set, self.epoch, main_metrics)

        if self.verbose:
            logging.info(f"\n{table}")
        else:
            # Log to file only if console is suppressed
            # We can force the message to be logged by temporarily ensuring handlers are set correctly
            # But simpler approach: logging.info still goes to FileHandler.
            # We need to mute StdoutHandler specifically.
            self.stdout_handler.setLevel(logging.WARNING)
            logging.info(f"\n{table}")
            self.stdout_handler.setLevel(logging.INFO)

        # Log sample counts for ranking metrics
        total_classified_samples = sum(self.batch_sizes)

        # Calculate total TP, FP, FN for class 1 (positive edges)
        tp_class1 = self.conf_mat_tp[1]
        fp_class1 = self.conf_mat_fp[1]
        fn_class1 = self.conf_mat_fn[1]

        sample_info = (
            f"Total samples: {total_classified_samples:,} "
            f"(Class 1: TP={tp_class1}, FP={fp_class1}, FN={fn_class1}) | "
            f"MAP: {total_map_samples:,} edges"
        )
        if self.verbose:
            logging.info(sample_info)
        else:
            self.stdout_handler.setLevel(logging.WARNING)
            logging.info(sample_info)
            self.stdout_handler.setLevel(logging.INFO)

        # Per-class metrics table
        class_headers = ["Class", "Precision", "Recall", "F1"]
        class_rows = [
            [cl, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]
            for cl, m in class_metrics.items()
        ]
        class_table = format_table(class_headers, class_rows, title="Per-Class Metrics")
        if self.verbose:
            logging.info(f"\n{class_table}")
        else:
            self.stdout_handler.setLevel(logging.WARNING)
            logging.info(f"\n{class_table}")
            self.stdout_handler.setLevel(logging.INFO)

        # Log to WandB
        if wandb.run:
            try:
                # Base metrics with set prefix for organization
                prefix = f"{self.set.lower()}/"

                # Increment global step for epoch-level logging
                self.global_step += 1

                # Get current timestamp
                timestamp = datetime.datetime.now().isoformat()

                wandb.log(
                    {
                        f"{prefix}epoch": self.epoch,
                        f"{prefix}loss": mean_loss,
                        f"{prefix}macro_precision": precision,
                        f"{prefix}macro_recall": recall,
                        f"{prefix}macro_f1": f1,
                        f"{prefix}map": epoch_MAP,
                        f"{prefix}auc_roc": epoch_AUC,
                        f"{prefix}timestamp": timestamp,
                        f"{prefix}total_samples": total_classified_samples,
                        f"{prefix}map_samples": total_map_samples,
                        # Per-class metrics
                        **{
                            f"{prefix}class_{cl}_precision": m["precision"]
                            for cl, m in class_metrics.items()
                        },
                        **{
                            f"{prefix}class_{cl}_recall": m["recall"]
                            for cl, m in class_metrics.items()
                        },
                        **{f"{prefix}class_{cl}_f1": m["f1"] for cl, m in class_metrics.items()},
                        # Confusion matrix for class 1
                        f"{prefix}tp_class1": tp_class1,
                        f"{prefix}fp_class1": fp_class1,
                        f"{prefix}fn_class1": fn_class1,
                    },
                    step=self.global_step,
                )
            except Exception as e:
                logging.warning(f"Failed to log to WandB: {e}")

        return eval_measure

    def compute_and_log_phase_graph_metrics(
        self, num_nodes: int, phase_idx: int, snapshot_idx: int
    ) -> Dict[str, float]:
        """Compute and log graph structure metrics at the end of a training phase.

        This method should be called at the end of each incremental training phase
        to compute structural metrics on the predicted graph from the last epoch's
        TEST predictions.

        Args:
            num_nodes: Total number of nodes in the graph
            phase_idx: Current phase index
            snapshot_idx: Index of the snapshot being tested

        Returns:
            Dict with all graph metrics, or empty dict if no predictions accumulated
        """
        # Check if we have accumulated predictions
        if not hasattr(self, "pred_edges_idx") or len(self.pred_edges_idx) == 0:
            logging.warning("No TEST predictions accumulated for graph metrics")
            return {}

        # Import graph_metrics here to avoid circular import
        try:
            import graph_metrics

            # Concatenate edge indices and values
            all_edge_idx = torch.cat(self.pred_edges_idx, dim=1)  # [2, total_edges]
            all_edge_vals = torch.cat(self.pred_edges_vals, dim=0)  # [total_edges]
            torch.cat(self.pred_edge_probs, dim=0)  # [total_edges]

            # Compute metrics
            metrics = graph_metrics.GraphMetricsCalculator.compute_all_metrics(
                all_edge_idx, all_edge_vals, num_nodes, threshold=0.5
            )

            # Log to console (ASCII table)
            self.log_graph_metrics_table(metrics, phase_idx, snapshot_idx)

            # Log to WandB
            if wandb.run:
                try:
                    # Use global_step to ensure monotonically increasing steps
                    # phase_idx is included as a metric for tracking which phase this is
                    self.global_step += 1
                    timestamp = datetime.datetime.now().isoformat()
                    wandb.log(
                        {
                            "graph/avg_degree": metrics["average_degree"],
                            "graph/avg_shortest_path": metrics["average_shortest_path_length"],
                            "graph/modularity": metrics["modularity"],
                            "graph/avg_clustering": metrics["average_clustering"],
                            "graph/num_communities": metrics["num_communities"],
                            "graph/gcc_ratio": metrics["gcc_ratio"],
                            "graph/gcc_size": metrics["gcc_size"],
                            "graph/total_nodes": metrics["total_nodes"],
                            "graph/phase": phase_idx,  # For tracking which phase this is
                            "graph/snapshot": snapshot_idx,
                            "graph/epoch": self.epoch,
                            "graph/timestamp": timestamp,
                        },
                        step=self.global_step,
                    )
                except Exception as e:
                    logging.warning(f"Failed to log graph metrics to WandB: {e}")

            # Note: We do NOT clear the buffers here
            # The trainer needs them in _save_phase_graphs to save graph files
            # Buffers will be automatically cleared at the start of the next TEST epoch

            return metrics

        except Exception as e:
            logging.error(f"Failed to compute graph metrics: {e}")
            return {}

    def log_graph_metrics_table(
        self, metrics: Dict[str, float], phase_idx: int, snapshot_idx: int
    ) -> None:
        """Log graph metrics as ASCII table to console."""
        headers = [
            "Avg Degree",
            "Avg Short Path",
            "Modularity",
            "Avg Cluster",
            "Communities",
            "GCC Ratio",
        ]

        gcc_ratio = (
            metrics["gcc_size"] / metrics["total_nodes"] if metrics["total_nodes"] > 0 else 0
        )

        values = [
            f"{metrics['average_degree']:.4f}",
            f"{metrics['average_shortest_path_length']:.4f}"
            if metrics["average_shortest_path_length"] is not None
            else "N/A",
            f"{metrics['modularity']:.4f}" if metrics["modularity"] is not None else "N/A",
            f"{metrics['average_clustering']:.4f}",
            str(metrics["num_communities"]),
            f"{gcc_ratio:.4f}",
        ]

        table = format_table(
            headers, [values], title=f"Graph Metrics | Phase {phase_idx} | Snapshot {snapshot_idx}"
        )
        logging.info(f"\n{table}")

    def _save_predictions_to_edg(self):
        """Save TEST predictions to .edg file."""
        # Check if we have accumulated predictions
        if not hasattr(self, "pred_edges_idx") or len(self.pred_edges_idx) == 0:
            logging.info("No TEST predictions to save")
            return

        try:
            # Concatenate all predictions from batches
            all_edge_idx = torch.cat(self.pred_edges_idx, dim=1)  # [2, total_edges]
            all_edge_probs = torch.cat(self.pred_edge_probs, dim=0)  # [total_edges]
            # all_edge_vals = torch.cat(self.pred_edges_vals, dim=0)  # [total_edges]
            all_edge_labels = torch.cat(
                self.pred_edges_labels, dim=0
            )  # [total_edges] - ground truth

            # Convert to numpy for easier handling
            src_nodes = all_edge_idx[0].cpu().numpy()
            dst_nodes = all_edge_idx[1].cpu().numpy()
            probs = all_edge_probs.cpu().numpy()
            labels = all_edge_labels.cpu().numpy()

            # Create predictions directory if it doesn't exist
            pred_dir = Path("log/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with epoch and timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{self.set.lower()}_epoch{self.epoch}_{timestamp}.edg"
            filepath = pred_dir / filename

            # Save all predictions (both positive and negative) with scores
            with open(filepath, "w") as f:
                f.write("# Source\tTarget\tProbability\tLabel\n")
                for src, dst, prob, label in zip(src_nodes, dst_nodes, probs, labels):
                    f.write(f"{src}\t{dst}\t{prob:.6f}\t{label}\n")

            num_positive = (labels == 1).sum()
            num_predicted_positive = (probs > 0.5).sum()
            num_predicted_negative = (probs <= 0.5).sum()

            logging.info(f"Saved predictions to {filepath}")
            logging.info(
                f"Total edges: {len(probs):,} | "
                f"Positive labels: {num_positive:,} | "
                f"Predicted positive: {num_predicted_positive:,} | "
                f"Predicted negative: {num_predicted_negative:,}"
            )

            # Store filepath for potential reuse (e.g., loading best epoch predictions)
            self.last_predictions_filepath = str(filepath)

        except Exception as e:
            logging.error(f"Failed to save predictions to .edg: {e}")

    def get_MAP(self, predictions, true_classes, do_softmax=False):
        """Calculate Mean Average Precision (MAP).

        Args:
            predictions: Model predictions.
            true_classes: Ground truth labels.
            do_softmax: Whether to apply softmax to predictions.

        Returns:
            float: Mean Average Precision value.
        """
        probs = torch.softmax(predictions, dim=1)[:, 1] if do_softmax else predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return average_precision_score(true_classes_np, predictions_np)

    def get_AUC_ROC(self, predictions, true_classes, do_softmax=False):
        """Calculate AUC-ROC on sampled negatives.

        This computes AUC-ROC on the evaluated sample set (positive edges + sampled negatives),
        NOT on all possible node pairs. This is the standard "sampling-based AUC" approach
        used in link prediction literature for computational efficiency.

        The metric measures how well the model distinguishes between:
        - Real new edges (positive samples)
        - Randomly sampled non-existent edges (negative samples with 1:1 ratio)

        Note: This is NOT the "global AUC" over all N*(N-1)/2 possible pairs,
        which would be computationally prohibitive for large graphs.

        Args:
            predictions: Model predictions (probabilities for positive class).
            true_classes: Ground truth labels (0=negative, 1=positive).
            do_softmax: Whether to apply softmax to predictions.

        Returns:
            float: AUC-ROC score on the sampled edge set.
        """
        probs = torch.softmax(predictions, dim=1)[:, 1] if do_softmax else predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return roc_auc_score(true_classes_np, predictions_np)

    def eval_predicitions(self, predictions, true_classes, num_classes=2):
        """Evaluate predictions and compute confusion matrix metrics.

        Args:
            predictions: Model predictions (logits).
            true_classes: Ground truth labels.
            num_classes: Number of classes.

        Returns:
            tuple: (error rate, confusion matrix per class)
        """
        predicted_classes = predictions.argmax(dim=1)
        failures = (predicted_classes != true_classes).sum(dtype=torch.float)
        error = failures / predictions.size(0)

        conf_mat_per_class = SimpleNamespace()
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

    def eval_predicitions_at_k(self, predictions, true_classes, num_classes=2, k=10):
        """Evaluate predictions at top-k.

        Args:
            predictions: Model predictions (logits).
            true_classes: Ground truth labels.
            num_classes: Number of classes.
            k: Top-k value.

        Returns:
            Confusion matrix per class for top-k predictions.
        """
        conf_mat_per_class = SimpleNamespace()
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

    def calc_eval_measures_per_class(self, tp, fn, fp, class_id):
        """Calculate precision, recall, and F1-score for a specific class.

        Args:
            tp: True positives (dict or scalar).
            fn: False negatives (dict or scalar).
            fp: False positives (dict or scalar).
            class_id: Target class ID.

        Returns:
            tuple: (precision, recall, F1-score)
        """
        if type(tp) is dict:
            tp_sum = tp[class_id].item()
            fn_sum = fn[class_id].item()
            fp_sum = fp[class_id].item()
        else:
            tp_sum = tp.item()
            fn_sum = fn.item()
            fp_sum = fp.item()

        if tp_sum == 0:
            p = 0.0 if fp_sum > 0 else 0.0
            r = 0.0 if fn_sum > 0 else 0.0
            f1 = 0.0
            return p, r, f1

        p = tp_sum * 1.0 / (tp_sum + fp_sum)
        r = tp_sum * 1.0 / (tp_sum + fn_sum)
        f1 = 2.0 * (p * r) / (p + r) if p + r > 0 else 0
        return p, r, f1

    def calc_epoch_metric(self, batch_sizes, metric_val):
        """Calculate weighted average metric across batches.

        Args:
            batch_sizes: Sizes of each batch.
            metric_val: Metric values for each batch.

        Returns:
            float: Weighted average metric.
        """
        batch_sizes = torch.tensor(batch_sizes, dtype=torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum() / batch_sizes.sum()

        return epoch_metric_val.detach().item()
