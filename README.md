# BigData-Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Temporal link prediction of risky users in social networks using Evolving Graph Convolutional Networks (EvolveGCN). The model is trained incrementally on temporal snapshots of the Gab social network, and used to predict connections for synthetic injected users.

## Setup

Requires Python 3.12 and a CUDA-capable GPU.

```bash
uv sync
```

## Running Experiments

```bash
# Train a model
uv run src/run_exp.py --config_file experiments/gab_h_c1_1_150.yaml

# Run inference on synthetic users
uv run src/infer_synthetic_incremental.py \
    --model_path log/gab_h_c1_1_50/checkpoint_phase_4_best.pth.tar \
    --config log/gab_h_c1_1_50/gab_h_c1_1_150.yaml \
    --output_dir gab_h_c1_1_50
```

## Experiment Configurations

Four main configurations varying learning rate and negative sampling ratio:

| Config         | Learning Rate | Neg. Mult. |
|----------------|---------------|------------|
| `gab_h_c1_1`   | 0.001         | 1          |
| `gab_h_c1_2`   | 0.001         | 2          |
| `gab_h_c2_1`   | 0.0005        | 1          |
| `gab_h_c2_2`   | 0.0005        | 2          |

All configs use EGCN-H, 768-dim BERT node features, MAP as the primary validation metric, and W&B for experiment tracking.

## Project Structure

```
├── pyproject.toml                  <- Package metadata and dependencies
├── README.md
│
├── data/
│   ├── raw/                        <- Original Gab social network data
│   ├── interim/                    <- Intermediate processing outputs
│   ├── processed/                  <- Finalized graph snapshots
│   ├── embeddings/                 <- BERT node feature embeddings
│   └── external/                   <- External datasets (Twitter, etc.)
│
├── experiments/                    <- YAML experiment configurations
│   ├── gab_h_c1_1_150.yaml
│   ├── gab_h_c1_2_150.yaml
│   ├── gab_h_c2_1_150.yaml
│   └── gab_h_c2_2_150.yaml
│
├── graphs/                         <- Temporal graph snapshots (phases 0-4, train/test splits)
│
├── src/
│   ├── run_exp.py                  <- Experiment entry point
│   ├── trainer.py                  <- Training, validation, and evaluation loop
│   ├── GabDataset.py               <- Gab dataset loader with temporal edges
│   ├── LinkPrediction.py           <- Link prediction tasker
│   ├── splitter.py                 <- Train/dev/test temporal splits
│   ├── logger.py                   <- Metrics logging (MAP, AUC, F1, precision, recall)
│   ├── infer_synthetic_incremental.py  <- Batch-incremental inference for synthetic users
│   ├── graph_metrics.py            <- GPU-accelerated graph metrics (cuDF/cuGraph)
│   ├── taskers_utils.py            <- Graph manipulation utilities
│   ├── utils.py                    <- Miscellaneous utilities
│   │
│   ├── modeling/
│   │   ├── egcn_h.py               <- EGCN-H: Evolving GCN with history
│   │   ├── egcn_o.py               <- EGCN-O: Evolving GCN original variant
│   │   └── MLP.py                  <- Classification head for link prediction
│   │
│   ├── embedding_analysis/         <- Embedding visualization and comparison tools
│   │   ├── run_all.py              <- Batch runner for all analyses
│   │   ├── analyze_gcn_embeddings.py
│   │   ├── analyze_multi_model_gcn.py
│   │   ├── analyze_nearest_neighbors.py
│   │   ├── analyze_real_embeddings.py
│   │   ├── analyze_synthetic_embeddings.py
│   │   ├── analyze_user_embeddings.py
│   │   ├── compare_embeddings.py
│   │   ├── run_bert_only.py
│   │   └── visualize_embeddings.py <- t-SNE / UMAP visualization
│   │
│   └── EvolveGCNORIGINAL/          <- Reference IBM EvolveGCN implementation, adapted to the dataset
│
├── scripts/                        <- Data processing and validation scripts
│   ├── create_edge_lists.py
│   ├── check_edge_duplicates.py
│   ├── check_raw_data.py
│   ├── analyze_test_leakage.py
│   ├── audit_sampling.py
│   ├── verify_edge_selection.py
│   ├── verify_incrementality_edges.py
│   ├── verify_negative_sampling.py
│   ├── embedding_drift.py          <- Drift between real and synthetic embeddings
│   └── visualize_embeddings.py
│
├── notebooks/
│   ├── 1-DescribeData.ipynb        <- Temporal graph exploration and node statistics
│   └── 2-DataQuality.ipynb         <- Data completeness, outliers, and consistency checks
│
│
├── reports/
│   ├── main.tex / main.pdf         <- Full project report (CRISP-DM structure)
│   ├── main.bib                    <- Bibliography
│   ├── figures/
│   └── presentation/
│
└── references/                     <- Data dictionaries and reference materials
```

--------

<p align="center">🚀</p>