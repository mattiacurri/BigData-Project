"""Entry point and helpers to build dataset, models and run experiments."""

import contextlib
import random

import Cross_Entropy as ce
import egcn_h
import egcn_o

# taskers
import link_pred_tasker as lpt

# models
import models as mls
import numpy as np
import torch

import splitter as sp
import trainer as tr
import utils as u


def random_param_value(param, param_min, param_max, type="int"):
    """Return `param` or sample a random value between min/max according to `type`."""
    if str(param) is None or str(param).lower() == "none":
        if type == "int":
            return random.randrange(param_min, param_max + 1)
        if type == "logscale":
            interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        return random.uniform(param_min, param_max)
    return param


# def build_random_hyper_params(args):
# 	if args.model == 'all':
# 		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
# 		args.model=model_types[args.rank]
# 	elif args.model == 'all_nogcn':
# 		model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
# 		args.model=model_types[args.rank]
# 	elif args.model == 'all_noegcn3':
# 		model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
# 		args.model=model_types[args.rank]
# 	elif args.model == 'all_nogruA':
# 		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
# 		args.model=model_types[args.rank]
# 		args.model=model_types[args.rank]
# 	elif args.model == 'saveembs':
# 		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
# 		args.model=model_types[args.rank]

# 	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
# 	# args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

# 	if args.model == 'gcn':
# 		args.num_hist_steps = 0
# 	else:
# 		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

# 	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
# 	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
# 	if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
# 		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
# 	else:
# 		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
# 	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
# 	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
# 		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
# 	else:
# 		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
# 	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
# 	return args


def build_dataset(args):
    """Construct and return the dataset specified by `args`."""
    import gab

    return gab.Gab(args)


def build_tasker(args, dataset):
    """Return a tasker instance for the configured `args` and `dataset`."""
    return lpt.Link_Pred_Tasker(args, dataset)


def build_gcn(args, tasker):
    """Build the requested GCN variant according to `args.model`."""
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node
    if args.model == "egcn_h":
        return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
    if args.model == "egcn_o":
        return egcn_o.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
    raise NotImplementedError("need to finish modifying the models")


def build_classifier(args, tasker):
    """Create classifier head using `tasker.num_classes` and `args` settings."""
    # if 'node_cls' == args.task or 'static_node_cls' == args.task:
    # 	mult = 1
    # else:
    mult = 2
    # if 'gru' in args.model or 'lstm' in args.model:
    # 	in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
    # elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
    # 	in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
    # else:
    in_feats = args.gcn_parameters["layer_2_feats"] * mult

    return mls.Classifier(args, in_features=in_feats, out_features=tasker.num_classes).to(
        args.device
    )


if __name__ == "__main__":
    parser = u.create_parser()
    args = u.parse_args(parser)

    global rank, wsize, use_cuda
    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    args.device = "cpu"
    if args.use_cuda:
        args.device = "cuda"
    print("use CUDA:", args.use_cuda, "- device:", args.device)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force deterministic algorithms for complete reproducibility
    # This may slightly reduce performance but ensures identical results
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)

    # # Assign the requested random hyper parameters
    # args = build_random_hyper_params(args)

    # build the dataset
    dataset = build_dataset(args)
    # build the tasker
    tasker = build_tasker(args, dataset)
    # build the splitter
    splitter = sp.splitter(args, tasker)
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

    trainer.train()
