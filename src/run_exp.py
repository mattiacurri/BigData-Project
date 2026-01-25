"""Experiment runner for graph neural network models.

This module orchestrates the training pipeline for GCN-based models on temporal graph data.
It handles model construction, hyperparameter randomization, dataset loading, and trainer initialization.
"""

import random

import numpy as np
import torch

# datasets
import cross_entropy as ce

# losses
import GabDataset as ds

# taskers
import link_pred_tasker as lpt
import modeling.egcn_h as egcn_h
import modeling.egcn_o as egcn_o

# models
import models as mls
import splitter as sp
import trainer as tr
import utils as u


def random_param_value(param, param_min, param_max, type="int"):
    """Generate a random parameter value within specified bounds.

    Args:
            param: The parameter value. If None/'none', a random value is generated.
            param_min: Minimum parameter value.
            param_max: Maximum parameter value.
            type: Type of sampling - 'int' (uniform), 'logscale' (log-uniform), or 'float' (uniform).

    Returns:
            The parameter value (random if param is None, otherwise the param itself).
    """
    if str(param) is None or str(param).lower() == "none":
        if type == "int":
            return random.randrange(param_min, param_max + 1)
        elif type == "logscale":
            interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param


def build_random_hyper_params(args):
    """Randomize and build hyperparameters for the model.

    Args:
            args: Configuration namespace containing parameter bounds.

    Returns:
            args: Updated configuration with randomized hyperparameters.
    """
    args.learning_rate = random_param_value(
        args.learning_rate, args.learning_rate_min, args.learning_rate_max, type="logscale"
    )
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

    if args.model == "gcn":
        args.num_hist_steps = 0
    else:
        args.num_hist_steps = random_param_value(
            args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type="int"
        )

    args.gcn_parameters["feats_per_node"] = random_param_value(
        args.gcn_parameters["feats_per_node"],
        args.gcn_parameters["feats_per_node_min"],
        args.gcn_parameters["feats_per_node_max"],
        type="int",
    )
    args.gcn_parameters["layer_1_feats"] = random_param_value(
        args.gcn_parameters["layer_1_feats"],
        args.gcn_parameters["layer_1_feats_min"],
        args.gcn_parameters["layer_1_feats_max"],
        type="int",
    )
    if (
        args.gcn_parameters["layer_2_feats_same_as_l1"]
        or args.gcn_parameters["layer_2_feats_same_as_l1"].lower() == "true"
    ):
        args.gcn_parameters["layer_2_feats"] = args.gcn_parameters["layer_1_feats"]
    else:
        args.gcn_parameters["layer_2_feats"] = random_param_value(
            args.gcn_parameters["layer_2_feats"],
            args.gcn_parameters["layer_1_feats_min"],
            args.gcn_parameters["layer_1_feats_max"],
            type="int",
        )
    args.gcn_parameters["lstm_l1_feats"] = random_param_value(
        args.gcn_parameters["lstm_l1_feats"],
        args.gcn_parameters["lstm_l1_feats_min"],
        args.gcn_parameters["lstm_l1_feats_max"],
        type="int",
    )
    if (
        args.gcn_parameters["lstm_l2_feats_same_as_l1"]
        or args.gcn_parameters["lstm_l2_feats_same_as_l1"].lower() == "true"
    ):
        args.gcn_parameters["lstm_l2_feats"] = args.gcn_parameters["lstm_l1_feats"]
    else:
        args.gcn_parameters["lstm_l2_feats"] = random_param_value(
            args.gcn_parameters["lstm_l2_feats"],
            args.gcn_parameters["lstm_l1_feats_min"],
            args.gcn_parameters["lstm_l1_feats_max"],
            type="int",
        )
    args.gcn_parameters["cls_feats"] = random_param_value(
        args.gcn_parameters["cls_feats"],
        args.gcn_parameters["cls_feats_min"],
        args.gcn_parameters["cls_feats_max"],
        type="int",
    )
    return args


def build_gcn(args, tasker):
    """Build the GCN model based on configuration.

    Args:
            args: Configuration namespace specifying model type and parameters.
            tasker: Tasker object containing feature dimensionality.

    Returns:
            Compiled GCN model on the specified device.

    Raises:
            AssertionError: If LSTM/GRU models are used without sufficient history.
            NotImplementedError: If the specified model is not recognized.
    """
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node
    if args.model == "gcn":
        return mls.Sp_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    elif args.model == "skipgcn":
        return mls.Sp_Skip_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    elif args.model == "skipfeatsgcn":
        return mls.Sp_Skip_NodeFeats_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    else:
        assert args.num_hist_steps > 0, "more than one step is necessary to train LSTM"
        if args.model == "lstmA":
            return mls.Sp_GCN_LSTM_A(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == "gruA":
            return mls.Sp_GCN_GRU_A(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == "lstmB":
            return mls.Sp_GCN_LSTM_B(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == "gruB":
            return mls.Sp_GCN_GRU_B(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == "egcn_h":
            return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
        elif args.model == "egcn_o":
            return egcn_o.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
        else:
            raise NotImplementedError(
                f"{args.model} not implemented. Choose among: gcn, skipgcn, skipfeatsgcn, lstmA, lstmB, gruA, gruB, egcn_h, skipfeatsegcn_h, egcn_o"
            )


def build_classifier(args, tasker):
    """Build the classification layer.

    Args:
            args: Configuration namespace.
            tasker: Tasker object containing number of classes.

    Returns:
            Classifier module on the specified device.
    """
    mult = 2  # link prediction, classifier input [embedding_node_u || embedding_node_v]
    if "gru" in args.model or "lstm" in args.model:
        in_feats = args.gcn_parameters["lstm_l2_feats"] * mult
    elif args.model == "skipfeatsgcn" or args.model == "skipfeatsegcn_h":
        in_feats = (
            args.gcn_parameters["layer_2_feats"] + args.gcn_parameters["feats_per_node"]
        ) * mult
    else:
        in_feats = args.gcn_parameters["layer_2_feats"] * mult

    return mls.Classifier(args, in_features=in_feats, out_features=tasker.num_classes).to(
        args.device
    )


if __name__ == "__main__":
    parser = u.create_parser()
    args = u.parse_args(parser)

    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    if not args.use_cuda:
        raise ValueError(
            "GPU is required to run this code. Just to avoid long training times on CPU."
        )
    args.device = "cuda" if args.use_cuda else "cpu"
    print("use CUDA:", args.use_cuda, "- device:", args.device)

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed

    # Assign the requested random hyper parameters
    # if 'none' is specified for a parameter, a random value will be sampled
    # if a value is specified, that value will be used
    args = build_random_hyper_params(args)

    # build the dataset
    dataset = ds.GabDataset(args)
    # build the tasker
    tasker = lpt.Link_Pred_Tasker(args, dataset)

    # if args.incremental:
    # print("\n" + "=" * 60)
    # print("INCREMENTAL TRAINING MODE")
    # print("=" * 60 + "\n")

    # Set default finetune_epochs if not specified
    if not hasattr(args, "finetune_epochs") or args.finetune_epochs is None:
        args.finetune_epochs = max(args.num_epochs // 2, 5)
        print(f"Using default finetune_epochs: {args.finetune_epochs}")
    else:
        args.finetune_epochs = int(args.finetune_epochs)

    # build the incremental splitter
    splitter = sp.incremental_splitter(args, tasker)
    # else:
    #     # build the standard splitter
    #     splitter = sp.splitter(args, tasker)

    # build the models
    gcn = build_gcn(args, tasker)
    classifier = build_classifier(args, tasker)
    # build a loss
    cross_entropy = ce.Cross_Entropy(args, dataset).to(args.device)

    # trainer
    trainer = tr.Trainer(
        args,
        splitter=splitter,
        gcn=gcn,
        classifier=classifier,
        comp_loss=cross_entropy,
        dataset=dataset,
        num_classes=tasker.num_classes,
    )

    # if args.incremental:
    trainer.train_incremental()
    # else:
    # trainer.train()
