"""Utilities for loading and computing graph-level metrics (GPU-accelerated)."""

import gzip
import json
import os
import urllib.request

import cudf
import cugraph
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================

DATASETS = [
    {
        "name": "ego-Facebook",
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "filename": "facebook_combined.txt.gz",
        "directed": False,
        "description": "Facebook Social Circles",
    },
    {
        "name": "ego-Twitter",
        "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
        "filename": "twitter_combined.txt.gz",
        "directed": True,
        "description": "Twitter Social Circles",
    },
    {
        "name": "ego-Gplus",
        "url": "https://snap.stanford.edu/data/gplus_combined.txt.gz",
        "filename": "gplus_combined.txt.gz",
        "directed": True,
        "description": "Google+ Social Circles",
    },
]

# Modifica con il percorso corretto del tuo file Gab
GAB_DATASET = {
    "name": "gab-jul25",
    "filepath": "../data/social_network.edg",
    "directed": True,
    "description": "Gab Social Network Jul 25",
}


def find_synthetic_datasets(data_dir="../data"):
    """Return a list of synthetic dataset descriptors found under `data_dir`."""
    datasets = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".edg") and "synthetic" in f.lower():
                datasets.append(
                    {
                        "name": f.replace(".edg", ""),
                        "filepath": os.path.join(data_dir, f),
                        "directed": True,
                        "description": f"Synthetic Network ({f})",
                    }
                )
    return datasets


SYNTHETIC_DATASETS = find_synthetic_datasets()

OUTPUT_DIR = "gpu_results_final"

# ==========================================
# FUNZIONI CORE
# ==========================================


def download_file(url: str, local_path: str):
    """Download `url` to `local_path` if missing or corrupted."""
    if not os.path.exists(local_path):
        print(f"Scaricando {url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
    elif local_path.endswith(".gz"):
        try:
            with gzip.open(local_path, "rb") as f:
                f.read(1)
        except:
            print(f"File corrotto, riscaricando...")
            os.remove(local_path)
            download_file(url, local_path)


def load_graph_gpu(filepath: str, directed: bool = False, expected_nodes: int = None):
    """Load a graph from `filepath`, renumber nodes to 0..N-1 and return GPU graph."""
    print(f" -> Caricamento: {os.path.basename(filepath)}")
    compression = "gzip" if filepath.endswith(".gz") else None

    try:
        # Legge solo le prime 2 colonne (ignora pesi)
        pdf = pd.read_csv(
            filepath,
            sep=r"\s+",
            names=["src", "dst"],
            usecols=[0, 1],
            compression=compression,
            comment="#",
            header=None,
            dtype=str,
        )

        # Converte a int
        pdf["src"] = pdf["src"].astype("int64")
        pdf["dst"] = pdf["dst"].astype("int64")

        # Trova tutti i nodi unici
        all_unique_nodes = sorted(set(pdf["src"]) | set(pdf["dst"]))
        num_unique = len(all_unique_nodes)

        # Crea mapping: ID originale -> 0, 1, 2, ...
        node_to_id = {n: i for i, n in enumerate(all_unique_nodes)}

        # Applica mapping
        pdf["src"] = pdf["src"].map(node_to_id).astype("int64")
        pdf["dst"] = pdf["dst"].map(node_to_id).astype("int64")

        # Passa a GPU
        gdf = cudf.from_pandas(pdf)

        # Crea grafo (gia' rinumerato, quindi renumber=False)
        G = cugraph.Graph(directed=directed)
        G.from_cudf_edgelist(
            gdf, source="src", destination="dst", renumber=False, store_transposed=True
        )

        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        print(f"    Grafo: {num_nodes} nodi, {num_edges} archi")

        # Per sintetici: usa expected_nodes (1000) se specificato
        actual_nodes = expected_nodes if expected_nodes else num_unique

        return G, gdf, actual_nodes

    except Exception as e:
        print(f"ERRORE caricamento: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def get_largest_connected_component_robust(G, edge_gdf, directed):
    """Estrae la LCC usando weakly_connected_components."""
    print(" -> Estrazione Componente Gigante (LCC)...")

    try:
        # Crea grafo non diretto per trovare componenti
        G_undirected = cugraph.Graph(directed=False)
        G_undirected.from_cudf_edgelist(edge_gdf, source="src", destination="dst", renumber=False)

        # Trova componenti connesse
        components = cugraph.weakly_connected_components(G_undirected)

        # Trova la componente più grande
        comp_counts = components["labels"].value_counts()
        largest_label = comp_counts.index[0]

        # Ottieni nodi della LCC
        lcc_nodes = components[components["labels"] == largest_label]["vertex"]

        # Filtra archi della LCC
        lcc_edges = edge_gdf[edge_gdf["src"].isin(lcc_nodes) & edge_gdf["dst"].isin(lcc_nodes)]

        # Crea grafo LCC
        G_lcc = cugraph.Graph(directed=directed)
        G_lcc.from_cudf_edgelist(
            lcc_edges,
            source="src",
            destination="dst",
            renumber=False,
            store_transposed=True,
        )

        print(f"    LCC: {G_lcc.number_of_nodes()} nodi")
        return G_lcc

    except Exception as e:
        print(f"    ERRORE LCC: {e}")
        import traceback

        traceback.print_exc()
        return None


def calculate_exact_avg_path_gpu(G, sample_size=None):
    """Calcola ASP usando BFS."""
    nodes_df = G.nodes().to_pandas()
    # Il nome della colonna potrebbe variare
    # Gestisce sia DataFrame che Series
    if hasattr(nodes_df, "columns"):
        col_name = nodes_df.columns[0] if len(nodes_df.columns) > 0 else "vertex"
        nodes_list = nodes_df[col_name].tolist()
    else:
        # Se è una Series, usa direttamente i valori
        nodes_list = nodes_df.tolist()

    if sample_size and len(nodes_list) > sample_size:
        import random

        nodes_list = random.sample(nodes_list, sample_size)

    total_distance = 0.0
    total_pairs = 0

    for start_node in tqdm(nodes_list, unit="node", ncols=80):
        df_dists = cugraph.bfs(G, start_node)
        distances = df_dists["distance"].to_pandas()

        # Escludi distanza 0 (nodo stesso) E valori sentinel (non raggiungibili)
        # I valori sentinel in cugraph sono tipicamente il max int o float
        valid = distances[(distances > 0) & (distances < 1e10)]
        total_distance += valid.sum()
        total_pairs += len(valid)

    if total_pairs == 0:
        return 0.0
    return total_distance / total_pairs


# ==========================================
# MAIN
# ==========================================


def analyze_dataset_precise(dataset, data_dir="../data"):
    """Compute precise graph metrics for `dataset` using GPU-accelerated methods.

    Returns a metrics dict including nodes, edges, average degree, modularity,
    average shortest path (LCC), and clustering coefficient when applicable.
    """
    filepath = dataset.get("filepath")
    if not filepath:
        filepath = os.path.join(data_dir, dataset["filename"])
        download_file(dataset.get("url"), filepath)

    if not os.path.exists(filepath):
        print(f"SKIP: File non trovato {filepath}")
        return None

    print(f"\n{'=' * 60}")
    print(f"ANALISI GPU: {dataset['description']}")
    print(f"{'=' * 60}")

    # 1. Carica
    # Per i dataset sintetici, ci aspettiamo 1000 nodi totali
    expected_nodes = 1000 if "synthetic" in dataset.get("name", "").lower() else None
    G, edge_gdf, actual_nodes = load_graph_gpu(filepath, dataset["directed"], expected_nodes)
    if G is None:
        return None

    metrics = {
        "dataset": dataset["description"],
        "nodes": actual_nodes,
        "edges": G.number_of_edges(),
        "directed": dataset["directed"],
    }

    # 2. Average Degree
    if dataset["directed"]:
        metrics["average_degree"] = float(metrics["edges"] / metrics["nodes"])
    else:
        metrics["average_degree"] = float((2 * metrics["edges"]) / metrics["nodes"])

    # 3. Modularity (Louvain)
    # TRUCCO: Louvain richiede grafo non diretto.
    # Se è diretto, creiamo una vista non diretta TEMPORANEA solo per Louvain.
    print(" -> Calcolo Modularity...")
    try:
        if dataset["directed"]:
            # Converti in non diretto solo per questo calcolo
            G_undirected = cugraph.Graph(directed=False)
            G_undirected.from_cudf_edgelist(
                edge_gdf, source="src", destination="dst", renumber=False
            )
            _, mod_score = cugraph.louvain(G_undirected)
            del G_undirected
        else:
            _, mod_score = cugraph.louvain(G)

        metrics["modularity"] = float(mod_score)
        print(f"    Modularity: {mod_score:.4f}")
    except Exception as e:
        print(f"    ERRORE Modularity: {e}")
        metrics["modularity"] = None

    # 4. Average Shortest Path (su LCC, con sampling 20% se LCC > 10000 nodi, Gab escluso)
    try:
        G_lcc = get_largest_connected_component_robust(G, edge_gdf, dataset["directed"])
        if G_lcc:
            lcc_size = G_lcc.number_of_nodes()
            is_special = any(x in dataset.get("name", "").lower() for x in ["gab", "synthetic"])
            sample_size = (
                lcc_size if is_special else (int(lcc_size * 0.2) if lcc_size > 10000 else lcc_size)
            )
            avg_path = calculate_exact_avg_path_gpu(G_lcc, sample_size=sample_size)
            metrics["average_shortest_path"] = float(avg_path)
            print(f"    Avg Path Length: {avg_path:.4f}")
        else:
            metrics["average_shortest_path"] = None
    except Exception as e:
        print(f"    ERRORE ASP: {e}")
        metrics["average_shortest_path"] = None

    # 5. Clustering Coefficient (solo per Gab e Synthetic)
    is_special = any(x in dataset.get("name", "").lower() for x in ["gab", "synthetic"])
    if is_special:
        print(" -> Calcolo Clustering Coefficient...")
        try:
            import networkx as nx

            # Usa compute_clustering_only che gestisce correttamente il formato
            cc_score = compute_clustering_only(dataset, data_dir)
            if cc_score is not None:
                metrics["clustering_coefficient"] = float(cc_score)
            else:
                metrics["clustering_coefficient"] = None
        except Exception as e:
            print(f"    ERRORE Clustering: {e}")
            metrics["clustering_coefficient"] = None
    else:
        metrics["clustering_coefficient"] = None

    return metrics


def save_metrics_txt(metrics, filepath):
    """Write `metrics` (dict) to `filepath` in a human-readable text format."""
    with open(filepath, "w") as f:
        f.write(f"Dataset: {metrics['dataset']}\n")
        f.write(f"Nodes: {metrics['nodes']}\n")
        f.write(f"Edges: {metrics['edges']}\n")
        f.write(f"Directed: {metrics['directed']}\n")
        f.write(f"Average Degree: {metrics['average_degree']:.4f}\n")
        f.write(f"Modularity: {metrics['modularity']:.4f}\n")
        if metrics["average_shortest_path"] is not None:
            f.write(f"Average Shortest Path: {metrics['average_shortest_path']:.4f}\n")
        else:
            f.write("Average Shortest Path: N/A\n")
        if metrics.get("clustering_coefficient") is not None:
            f.write(f"Clustering Coefficient: {metrics['clustering_coefficient']:.4f}\n")


def compute_clustering_only(dataset, data_dir="../data"):
    """Calcola clustering coefficient per grafi sintetici."""
    filepath = dataset.get("filepath")
    if not filepath:
        filepath = os.path.join(data_dir, dataset["filename"])

    if not os.path.exists(filepath):
        return None

    print(f" -> Calcolo Clustering Coefficient...")
    try:
        import networkx as nx

        compression = "gzip" if filepath.endswith(".gz") else None

        # Leggi solo prime 2 colonne
        pdf = pd.read_csv(
            filepath,
            sep=r"\s+",
            names=["src", "dst"],
            usecols=[0, 1],
            compression=compression,
            comment="#",
            header=None,
            dtype=str,
        )

        print(f"    Colonne dopo read_csv: {list(pdf.columns)}")

        # Converte e rimappa ID
        pdf["src"] = pdf["src"].astype("int64")
        pdf["dst"] = pdf["dst"].astype("int64")

        all_nodes = sorted(set(pdf["src"]) | set(pdf["dst"]))
        node_map = {n: i for i, n in enumerate(all_nodes)}
        pdf["src"] = pdf["src"].map(node_map).astype("int64")
        pdf["dst"] = pdf["dst"].map(node_map).astype("int64")

        # Crea grafo NetworkX
        nx_G = nx.from_pandas_edgelist(pdf, source="src", target="dst", create_using=nx.DiGraph())

        # Converti a non diretto per clustering
        nx_G = nx_G.to_undirected()

        cc_score = nx.average_clustering(nx_G)
        print(f"    Clustering Coefficient: {cc_score:.4f}")
        return float(cc_score)

    except Exception as e:
        print(f"    ERRORE Clustering: {e}")
        import traceback

        traceback.print_exc()
        return None

    print(f" -> Calcolo Clustering Coefficient...")
    try:
        import networkx as nx

        compression = "gzip" if filepath.endswith(".gz") else None

        # Leggi solo prime 2 colonne
        pdf = pd.read_csv(
            filepath,
            sep=r"\s+",
            names=["src", "dst"],
            usecols=[0, 1],
            compression=compression,
            comment="#",
            header=None,
            dtype=str,
        )

        # Converte e rimappa ID
        pdf["src"] = pdf["src"].astype("int64")
        pdf["dst"] = pdf["dst"].astype("int64")

        all_nodes = sorted(set(pdf["src"]) | set(pdf["dst"]))
        node_map = {n: i for i, n in enumerate(all_nodes)}
        pdf["src"] = pdf["src"].map(node_map).astype("int64")
        pdf["dst"] = pdf["dst"].map(node_map).astype("int64")

        # Crea grafo NetworkX
        nx_G = nx.from_pandas_edgelist(pdf, source="src", target="dst", create_using=nx.DiGraph())

        # Converti a non diretto per clustering
        nx_G = nx_G.to_undirected()

        cc_score = nx.average_clustering(nx_G)
        print(f"    Clustering Coefficient: {cc_score:.4f}")
        return float(cc_score)

    except Exception as e:
        print(f"    ERRORE Clustering: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # SNAP Datasets
    for ds in DATASETS:
        txt_path = os.path.join(OUTPUT_DIR, f"{ds['name']}_gpu.txt")
        json_path = os.path.join(OUTPUT_DIR, f"{ds['name']}_gpu.json")

        if os.path.exists(txt_path):
            print(f"SKIP {ds['name']}: già computato")
            with open(json_path, "r") as f:
                res = json.load(f)

            # Check if clustering is missing for special datasets
            is_special = any(x in ds.get("name", "").lower() for x in ["gab", "synthetic"])
            if is_special and res.get("clustering_coefficient") is None:
                print(f"   -> Clustering mancante, calcolo...")
                cc = compute_clustering_only(ds)
                if cc is not None:
                    res["clustering_coefficient"] = cc
                    with open(json_path, "w") as f:
                        json.dump(res, f, indent=2)
                    save_metrics_txt(res, txt_path)

            with open(txt_path, "r") as f:
                print(f.read())
            results.append(res)
            continue

        try:
            res = analyze_dataset_precise(ds)
            if res:
                results.append(res)
                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2)
                save_metrics_txt(res, txt_path)
        except Exception as e:
            print(f"SKIP {ds['name']}: {e}")

    # GAB Dataset
    txt_path = os.path.join(OUTPUT_DIR, "gab_gpu.txt")
    json_path = os.path.join(OUTPUT_DIR, "gab_gpu.json")
    if os.path.exists(txt_path):
        print(f"SKIP gab: già computato")
        with open(json_path, "r") as f:
            res = json.load(f)

        if res.get("clustering_coefficient") is None:
            print(f"   -> Clustering mancante, calcolo...")
            cc = compute_clustering_only(GAB_DATASET)
            if cc is not None:
                res["clustering_coefficient"] = cc
                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2)
                save_metrics_txt(res, txt_path)

        with open(txt_path, "r") as f:
            print(f.read())
        results.append(res)
    elif os.path.exists(GAB_DATASET["filepath"]):
        res_gab = analyze_dataset_precise(GAB_DATASET)
        if res_gab:
            results.append(res_gab)
            with open(json_path, "w") as f:
                json.dump(res_gab, f, indent=2)
            save_metrics_txt(res_gab, txt_path)
    else:
        print(f"Dataset GAB non trovato al percorso: {GAB_DATASET['filepath']}")

    # Synthetic Datasets
    for syn_ds in SYNTHETIC_DATASETS:
        safe_name = syn_ds["name"].replace(" ", "_").replace("-", "_")
        txt_path = os.path.join(OUTPUT_DIR, f"{safe_name}_gpu.txt")
        json_path = os.path.join(OUTPUT_DIR, f"{safe_name}_gpu.json")

        if os.path.exists(txt_path):
            print(f"SKIP {syn_ds['name']}: già computato")
            with open(json_path, "r") as f:
                res = json.load(f)

            if res.get("clustering_coefficient") is None:
                print(f"   -> Clustering mancante, calcolo...")
                cc = compute_clustering_only(syn_ds)
                if cc is not None:
                    res["clustering_coefficient"] = cc
                    with open(json_path, "w") as f:
                        json.dump(res, f, indent=2)
                    save_metrics_txt(res, txt_path)

            with open(txt_path, "r") as f:
                print(f.read())
            results.append(res)
        elif os.path.exists(syn_ds["filepath"]):
            res_syn = analyze_dataset_precise(syn_ds)
            if res_syn:
                results.append(res_syn)
                with open(json_path, "w") as f:
                    json.dump(res_syn, f, indent=2)
                save_metrics_txt(res_syn, txt_path)
        else:
            print(f"Dataset {syn_ds['name']} non trovato al percorso: {syn_ds['filepath']}")

    # Tabella finale
    print("\n\nRISULTATI FINALI:")
    for r in results:
        path_str = (
            f"{r['average_shortest_path']:.4f}"
            if r["average_shortest_path"] is not None
            else "N/A"
        )
        cc_str = (
            f", CC={r['clustering_coefficient']:.4f}"
            if r.get("clustering_coefficient") is not None
            else ""
        )
        print(
            f"{r['dataset']}: Deg={r['average_degree']:.2f}, Mod={r['modularity']:.4f}, Path={path_str}{cc_str}"
        )
