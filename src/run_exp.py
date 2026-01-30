"""Experiment runner for graph neural network models.

This module orchestrates the training pipeline for GCN-based models on temporal graph data.
It handles model construction, hyperparameter randomization, dataset loading, and trainer initialization.
"""

# parser
import argparse
import random

import numpy as np
import torch

# datasets
import GabDataset as ds

# taskers
from LinkPrediction import LinkPrediction

# models
import modeling.egcn_h as egcn_h
import modeling.egcn_o as egcn_o
import modeling.MLP as ClassifierHead
import splitter as sp
import trainer as tr
import utils as u

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_file",
        default="experiments/parameters_example.yaml",
        type=argparse.FileType(mode="r"),
        help="optional, yaml file containing parameters to be used, overrides command line parameters",
    )
    args = u.parse_args(parser)

    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    args.device = "cuda" if args.use_cuda else "cpu"
    print("use CUDA:", args.use_cuda, "- device:", args.device)

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed

    # deterministic cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build the dataset
    dataset = ds.GabDataset(args)
    # build the tasker
    tasker = LinkPrediction(args, dataset)

    # Set default finetune_epochs if not specified
    if not hasattr(args, "finetune_epochs") or args.finetune_epochs is None:
        args.finetune_epochs = max(args.num_epochs // 2, 5)
        print(f"Using default finetune_epochs: {args.finetune_epochs}")
    else:
        args.finetune_epochs = int(args.finetune_epochs)

    # build the incremental splitter
    if args.incremental:
        splitter = sp.IncrementalSplitter(args, tasker)
    else:
        splitter = sp.Splitter(args, tasker)

    # build the model
    # GCN
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = dataset.feats_per_node

    if args.model == "egcn_h":
        gcn = egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
    elif args.model == "egcn_o":
        gcn = egcn_o.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
    else:
        raise NotImplementedError(f"{args.model} not implemented. Choose among: egcn_h, egcn_o")

    # Classifier Head
    # 2 -> num_classes for link prediction
    # For EGCN models, use layer_2_feats; for LSTM-based models, use lstm_l2_feats
    if args.model in ["egcn_h", "egcn_o"]:
        gcn_output_dim = args.gcn_parameters["layer_2_feats"]
        print(f"Using EGCN model with output dimension: {gcn_output_dim}")
    else:
        gcn_output_dim = args.gcn_parameters["lstm_l2_feats"]
        print(f"Using LSTM-based model with output dimension: {gcn_output_dim}")

    print(f"Classifier input dimension (2*output_dim): {gcn_output_dim * 2}")
    classifier = ClassifierHead.MLP(args, in_features=gcn_output_dim * 2).to(args.device)

    # build a loss
    weights = torch.tensor(args.class_weights, dtype=torch.float).to(args.device)
    loss = torch.nn.CrossEntropyLoss(weight=weights)

    # trainer
    trainer = tr.Trainer(
        args,
        splitter=splitter,
        gcn=gcn,
        classifier=classifier,
        comp_loss=loss,
        dataset=dataset,
    )

    if args.incremental:
        trainer.train_incremental()
    else:
        trainer.train()
