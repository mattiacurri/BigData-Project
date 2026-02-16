# Embedding Analysis Scripts

## Script disponibili

| Script | Descrizione | Output |
|--------|-------------|--------|
| `analyze_synthetic_embeddings.py` | Statistiche embeddings BERT sintetici | `synthetic_embeddings_stats.json`, `synthetic_embeddings_distribution.png`, `synthetic_variance_per_dim.png` |
| `analyze_real_embeddings.py` | Statistiche embeddings BERT reali | `real_embeddings_stats.json`, `real_embeddings_distribution.png`, `real_variance_per_dim.png` |
| `compare_embeddings.py` | Confronto real vs synthetic | `embedding_comparison.json`, `embedding_comparison.png`, `embedding_variance_summary.png` |
| `analyze_user_embeddings.py` | Embedding per utente (media post) | `user_embedding_stats.json`, `user_embedding_analysis.png` |
| `analyze_nearest_neighbors.py` | Analisi nearest neighbors | `nearest_neighbors_stats.json`, `nearest_neighbors_analysis.png` |
| `analyze_gcn_embeddings.py` | Embeddings dopo GCN (singolo modello) | `gcn_embeddings_stats.json`, `gcn_embeddings_comparison.png`, `gcn_embeddings_distances.png`, `real_gcn_embeddings.npy`, `synth_gcn_embeddings.npy` |
| `visualize_embeddings.py` | PCA e t-SNE per BERT e GCN | `embeddings_visualization_bert.png`, `embeddings_visualization_gcn.png`, `visualization_stats.json` |
| `analyze_multi_model_gcn.py` | Confronto GCN multi-modello | `multi_model_gcn_comparison.json`, `multi_model_gcn_comparison.png` |
| `visualize_multi_model_gcn.py` | PCA e t-SNE per tutti i modelli GCN | `gcn_visualizations/{model_name}_gcn_visualization.png`, `all_models_gcn_overview.png`, `visualization_stats.json` |

## Esecuzione

```bash
cd src/embedding_analysis
uv run python run_all.py
```

Oppure singolo script:
```bash
uv run python analyze_synthetic_embeddings.py
```

## Output directory

Tutti i risultati sono salvati in `scripts/analysis/results/`.