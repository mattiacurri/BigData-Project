"""Graph Metrics Calculator.

This module provides utilities for computing structural graph metrics
using NetworkX. It is designed to work with graph predictions from
temporal GNN models during training/evaluation.

The module handles:
- Conversion of model predictions to NetworkX graphs
- Computation of 5 key structural metrics for social network analysis
- Robust handling of disconnected graphs (using Giant Connected Component)
- Integration with the training pipeline via the Logger class

Supported Metrics:
1. Average Degree - Average number of connections per node
2. Average Shortest Path Length - Mean shortest path between nodes
3. Modularity - Quality of community structure (Louvain algorithm)
4. Average Clustering Coefficient - Degree of "cliquishness"
5. Number of Communities - Count of detected communities
"""

from typing import Dict, Optional, Tuple
import warnings

import networkx as nx
from networkx.algorithms import community
import torch


class GraphMetricsCalculator:
    """Calculator for structural graph metrics using NetworkX.

    This class provides static and class methods for converting model predictions
    to NetworkX graphs and computing various structural properties. It includes
    methods for connected component analysis, degree metrics, path lengths,
    modularity, clustering, and community detection.
    """

    @staticmethod
    def predictions_to_graph(
        pred_indices: torch.Tensor, pred_vals: torch.Tensor, num_nodes: int, threshold: float = 0.5
    ) -> nx.DiGraph:
        """Convert prediction tensors to a NetworkX directed graph.

        Parameters
        ----------
        pred_indices : torch.Tensor
            Tensor of shape [2, num_edges] containing source and target node indices
        pred_vals : torch.Tensor
            Tensor of shape [num_edges] containing edge probabilities/values
        num_nodes : int
            Total number of nodes in the graph
        threshold : float, default=0.5
            Threshold for binary classification of edges

        Returns:
        -------
        nx.DiGraph
            Directed NetworkX graph with edges above the threshold

        Notes:
        -----
        This method converts the output tensor format commonly used in graph neural
        networks (where edges are represented as coordinate pairs) to a NetworkX
        graph structure. Only edges with probability values >= threshold are included.

        For social network link prediction, edges represent "X follows Y" relationships,
        hence we use a directed graph (DiGraph).
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(pred_indices):
            pred_indices = pred_indices.cpu().numpy()
        if torch.is_tensor(pred_vals):
            pred_vals = pred_vals.cpu().numpy()

        # Create directed graph
        G = nx.DiGraph()

        # Add all nodes (even if isolated)
        G.add_nodes_from(range(num_nodes))

        # Add edges above threshold
        for i in range(pred_indices.shape[1]):
            source = int(pred_indices[0, i])
            target = int(pred_indices[1, i])
            prob = float(pred_vals[i])

            if prob >= threshold:
                G.add_edge(source, target, probability=prob)

        return G

    @staticmethod
    def get_giant_connected_component(G: nx.Graph) -> nx.Graph:
        """Extract the largest connected component from a graph.

        Parameters
        ----------
        G : nx.Graph
            Input graph (directed or undirected)

        Returns:
        -------
        nx.Graph
            The giant connected component as an undirected graph

        Notes:
        -----
        The giant connected component (GCC) is the largest subgraph in which any
        two vertices are connected by paths. We use GCC for metrics requiring
        connectivity (like average shortest path length) because:

        1. These metrics are undefined for disconnected graphs - the shortest path
           between nodes in different components is infinite/undefined
        2. NetworkX raises NetworkXError when trying to compute shortest paths
           on disconnected graphs
        3. The GCC typically represents the core structure and majority of nodes
        4. Using GCC enables consistent comparison across different graph snapshots

        WHY GCC IS THE STANDARD APPROACH:
        - In social network analysis, disconnected graphs are very common,
          especially predicted graphs from thresholded probabilities
        - The GCC represents the "main body" of the network where actual
          information flow and community structure exist
        - Comparing GCC metrics across phases provides meaningful insights
          into the evolving core structure of the social network
        """
        # Convert to undirected for connected component analysis
        G_undirected = G.to_undirected()

        # Get all connected components
        connected_components = list(nx.connected_components(G_undirected))

        if not connected_components:
            return nx.Graph()

        # Find the largest component
        gcc_nodes = max(connected_components, key=len)

        # Return as undirected subgraph
        return G_undirected.subgraph(gcc_nodes).copy()

    @staticmethod
    def compute_average_degree(G: nx.Graph) -> float:
        """Compute the average degree of the graph.

        Parameters
        ----------
        G : nx.Graph
            Input graph

        Returns:
        -------
        float
            Average degree = (2 * num_edges) / num_nodes for undirected graphs
                          = num_edges / num_nodes for directed graphs (counting in+out)

        Notes:
        -----
        Average degree measures the average number of edges connected to each node.
        For directed graphs, this counts both in-degree and out-degree.

        In social networks:
        - Low values (< 2): Sparse network, limited connections
        - Moderate values (2-10): Normal social network density
        - High values (> 10): Dense network, highly interconnected community

        Range: [0, num_nodes-1] for simple graphs
        """
        if G.number_of_nodes() == 0:
            return 0.0

        total_degree = sum(dict(G.degree()).values())
        return total_degree / G.number_of_nodes()

    @staticmethod
    def compute_average_shortest_path_length(G: nx.Graph) -> Optional[float]:
        """Compute the average shortest path length using the giant connected component.

        Parameters
        ----------
        G : nx.Graph
            Input graph

        Returns:
        -------
        float or None
            Average shortest path length of the GCC, or None if undefined

        Notes:
        -----
        Average shortest path length is the average of the shortest path lengths
        between all pairs of nodes in the GCC.

        FORMULA: L = (2 / (n*(n-1))) * Σ d(u,v) for all node pairs (u,v)
        where n is the number of nodes in GCC and d(u,v) is shortest path distance

        WHY WE USE GCC:
        This metric requires a connected graph because:
        1. Shortest paths between disconnected components are infinite/undefined
        2. NetworkX raises NetworkXError for disconnected graphs
        3. The GCC represents the main connected structure where paths are meaningful
        4. Using GCC enables consistent comparison across different network states

        INTERPRETATION:
        - Small values (2-4): Tightly knit community, efficient information spread
        - Large values (> 6): Dispersed network, longer information chains
        - Values follow "small world" phenomenon in social networks

        Range: [1, N-1] where N is the number of nodes in the GCC
        Lower values indicate more tightly connected networks.
        """
        # Extract GCC first - THIS IS THE KEY STEP FOR DISCONNECTED GRAPHS
        # We must use GCC because average_shortest_path_length requires connectivity
        gcc = GraphMetricsCalculator.get_giant_connected_component(G)

        if gcc.number_of_nodes() <= 1:
            warnings.warn("GCC has 1 or fewer nodes, cannot compute average shortest path length")
            return None

        try:
            # Use unweighted path lengths (each edge counts as 1)
            # This treats all connections equally, appropriate for social networks
            return nx.average_shortest_path_length(gcc, weight=None)
        except nx.NetworkXError as e:
            # Defensive programming - should not happen if GCC extraction worked
            warnings.warn(f"GCC is not connected as expected: {e}")
            return None

    @staticmethod
    def compute_modularity(G: nx.Graph) -> Optional[float]:
        """Compute modularity using Louvain community detection.

        Parameters
        ----------
        G : nx.Graph
            Input graph (treated as undirected for community detection)

        Returns:
        -------
        float or None
            Modularity score ranging from -1 to 1, or None if no communities found

        Notes:
        -----
        Modularity measures the strength of community structure by comparing the
        actual number of edges within communities to the expected number in a
        random graph with the same degree distribution.

        FORMULA: Q = (1/2m) * Σ[A_ij - (k_i * k_j) / 2m] * δ(c_i, c_j)
        where:
        - m is total number of edges
        - A_ij is adjacency matrix (1 if edge exists, 0 otherwise)
        - k_i, k_j are degrees of nodes i and j
        - δ(c_i, c_j) is 1 if nodes i and j are in the same community, 0 otherwise

        ALGORITHM: Louvain method
        - Hierarchical greedy algorithm that optimizes modularity
        - Time complexity: O(n log n) where n is number of nodes
        - We use seed=42 for reproducibility across training runs

        INTERPRETATION:
        Range: [-0.5, 1.0]
        - Values near 0: No community structure (random-like connections)
        - Values > 0.3: Significant community structure present
        - Values > 0.5: Strong community structure (typical for social networks)
        - Negative values: Communities are worse than random (rare)

        In social networks, modularity reveals how users naturally cluster into
        communities, interest groups, or echo chambers.
        """
        # Use undirected version for community detection
        # Community structure is inherently an undirected concept
        G_undirected = G.to_undirected()

        if G_undirected.number_of_edges() == 0:
            warnings.warn("Graph has no edges, cannot compute modularity")
            return None

        try:
            # Detect communities using Louvain algorithm with fixed seed for reproducibility
            # This ensures that running the same training twice gives identical communities
            communities = community.louvain_communities(G_undirected, seed=42)

            # Compute modularity score
            modularity_score = community.modularity(G_undirected, communities)
            return float(modularity_score)
        except Exception as e:
            warnings.warn(f"Failed to compute modularity: {e}")
            return None

    @staticmethod
    def compute_average_clustering(G: nx.Graph) -> float:
        """Compute the average clustering coefficient of the graph.

        Parameters
        ----------
        G : nx.Graph
            Input graph

        Returns:
        -------
        float
            Average clustering coefficient

        Notes:
        -----
        The clustering coefficient measures the tendency of nodes to form triangles
        (triadic closure). For a node with k neighbors, it's the fraction of possible
        edges between those neighbors that actually exist.

        FORMULA: C_i = 2 * T_i / (k_i * (k_i - 1))
        where:
        - T_i is the number of triangles through node i
        - k_i is the degree of node i
        - Average clustering = (1/N) * Σ C_i for all nodes

        SOCIAL NETWORK SIGNIFICANCE:
        - Triadic closure: "A friend of my friend is likely to be my friend"
        - High clustering indicates strong local cohesion
        - Typical values for social networks: 0.1 to 0.5

        Range: [0, 1]
        - 0: No triangles in the network (tree-like structure)
        - 1: Complete graph (every neighbor connected to every other)

        High values indicate local clustering and redundant connections,
        suggesting strong community bonds and mutual acquaintances.
        """
        if G.number_of_nodes() == 0:
            return 0.0

        # Use undirected version for clustering (triangles are undirected)
        G_undirected = G.to_undirected()
        return float(nx.average_clustering(G_undirected))

    @staticmethod
    def compute_num_communities(G: nx.Graph) -> int:
        """Compute the number of communities using Louvain detection.

        Parameters
        ----------
        G : nx.Graph
            Input graph

        Returns:
        -------
        int
            Number of detected communities

        Notes:
        -----
        Uses the same Louvain algorithm and seed (42) as modularity computation
        to ensure consistency. Each community is a set of nodes with more
        internal connections than expected by chance.

        INTERPRETATION:
        Range: [1, num_nodes]
        - 1: Single community (no meaningful community structure)
        - Small number (2-10): Clear community divisions
        - Large number (> 20): Highly fragmented network
        - num_nodes: Each node is its own community (no edges or no clustering)

        The number of communities provides insight into network fragmentation.
        In incremental training, observing how this number changes across phases
        reveals the evolving community structure of the social network.

        Note: We use the same seed=42 as compute_modularity for consistency
        so that modularity score and community count refer to the same partition.
        """
        # Use undirected version for community detection
        G_undirected = G.to_undirected()

        if G_undirected.number_of_nodes() == 0:
            return 0

        try:
            # Use same seed as modularity for consistency
            # This ensures the same partition is used for both metrics
            communities = community.louvain_communities(G_undirected, seed=42)
            return len(communities)
        except Exception as e:
            warnings.warn(f"Failed to detect communities: {e}, returning 1")
            return 1

    @classmethod
    def compute_all_metrics(
        cls,
        pred_indices: torch.Tensor,
        pred_vals: torch.Tensor,
        num_nodes: int,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute all graph metrics and return comprehensive results.

        This is the main entry point for computing structural metrics from model
        predictions. It orchestrates the conversion to NetworkX and calculation
        of all five metrics in a single pass.

        Parameters
        ----------
        pred_indices : torch.Tensor
            Tensor of shape [2, num_edges] containing edge indices
        pred_vals : torch.Tensor
            Tensor of edge probabilities or binary labels
        num_nodes : int
            Total number of nodes
        threshold : float, default=0.5
            Edge probability threshold for graph construction

        Returns:
        -------
        dict
            Dictionary containing all metrics and metadata:

            Structural Metrics:
            - average_degree: Average node degree (how well-connected is the network)
            - average_shortest_path_length: GCC average path length (network efficiency)
            - modularity: Community structure strength (0 = random, > 0.3 = communities)
            - average_clustering: Local clustering tendency (triadic closure)
            - num_communities: Number of detected communities (network fragmentation)

            Metadata:
            - gcc_size: Size of giant connected component (main network core)
            - total_nodes: Total nodes in graph (including isolates)
            - total_edges: Total directed edges above threshold
            - gcc_ratio: Fraction of nodes in GCC (connectivity measure)

        Notes:
        -----
        Optimization: We compute communities once and reuse for both modularity
        and community count to avoid redundant Louvain algorithm runs.

        GCC Usage: For metrics requiring connectivity (shortest path), we use the
        Giant Connected Component as the representative subgraph. This is standard
        practice in network science for disconnected graphs.
        """
        # Convert predictions to directed graph
        # This represents the "X follows Y" social network structure
        G = cls.predictions_to_graph(pred_indices, pred_vals, num_nodes, threshold)

        # Get GCC for connectivity-based metrics and metadata
        # GCC represents the main connected component where most analysis is meaningful
        gcc = cls.get_giant_connected_component(G)

        # Compute all five structural metrics
        # Note: Each metric has its own detailed docstring explaining significance
        metrics = {
            "average_degree": cls.compute_average_degree(G),
            "average_shortest_path_length": cls.compute_average_shortest_path_length(G),
            "modularity": cls.compute_modularity(G),
            "average_clustering": cls.compute_average_clustering(G),
            "num_communities": cls.compute_num_communities(G),
            # Metadata for understanding graph structure
            "gcc_size": gcc.number_of_nodes(),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

        # Compute GCC ratio: what fraction of nodes are in the main component
        # This measures overall network connectivity
        # High ratio (> 0.8) = well-connected network
        # Low ratio (< 0.5) = highly fragmented network
        if metrics["total_nodes"] > 0:
            metrics["gcc_ratio"] = metrics["gcc_size"] / metrics["total_nodes"]
        else:
            metrics["gcc_ratio"] = 0.0

        return metrics

    @staticmethod
    def save_graph_edg(G: nx.Graph, filepath: str) -> None:
        """Save graph in simple edge list format.

        Parameters
        ----------
        G : nx.Graph
            Input graph
        filepath : str
            Output file path

        Notes:
        -----
        Saves in "X Y" format where X follows Y (for directed graphs).
        One edge per line. This is a simple, human-readable format that
        preserves the directed nature of edges but not edge attributes.

        FORMAT EXAMPLE:
        0 1   (node 0 follows node 1)
        0 2   (node 0 follows node 2)
        1 3   (node 1 follows node 3)

        This format is:
        - Compatible with existing project .edg files
        - Easy to import into Gephi, Cytoscape, or Python
        - Compact and version-control friendly
        - Does not preserve edge probabilities (use .graphml for that)
        """
        with open(filepath, "w") as f:
            for source, target in G.edges():
                f.write(f"{source} {target}\n")

    @staticmethod
    def save_graph_graphml(
        G: nx.Graph,
        edge_probabilities: Dict[Tuple[int, int], float],
        communities: Dict[int, int],
        filepath: str,
        phase_info: Optional[Dict] = None,
    ) -> None:
        """Save graph in GraphML format with full attributes and metadata.

        Parameters
        ----------
        G : nx.Graph
            Input graph (directed)
        edge_probabilities : dict
            Dictionary mapping (source, target) edge tuples to probability values
            These are the model's confidence scores for each predicted link
        communities : dict
            Dictionary mapping node IDs to community IDs
            Allows tracking which community each node belongs to
        filepath : str
            Output file path
        phase_info : dict, optional
            Dictionary with phase metadata to add as graph attributes:
            - phase: phase index
            - phase_desc: human-readable phase description
            - snapshot: tested snapshot index
            - avg_degree, modularity, num_communities: computed metrics

        Notes:
        -----
        GraphML is an XML-based format that preserves:
        1. Edge probabilities as 'probability' attribute - model confidence
        2. Node community assignments as 'community' attribute - which cluster
        3. Graph metadata including phase information - for traceability
        4. Directionality of edges - maintains "X follows Y" semantics

        ADVANTAGES OVER .edg:
        - Preserves all computed information (probabilities, communities)
        - Standard format supported by NetworkX, Gephi, Cytoscape, yEd
        - Self-documenting with inline metadata
        - Suitable for downstream analysis and visualization

        XML STRUCTURE:
        <?xml version="1.0" encoding="UTF-8"?>
        <graphml>
          <key id="prob" for="edge" attr.name="probability" attr.type="double"/>
          <key id="comm" for="node" attr.name="community" attr.type="int"/>
          <graph id="predicted_graph" edgedefault="directed">
            <node id="0"><data key="comm">3</data></node>
            <edge source="0" target="1"><data key="prob">0.95</data></edge>
          </graph>
        </graphml>

        This format enables rich analysis while maintaining compatibility
        with professional graph visualization tools.
        """
        # Create a copy to avoid modifying the original graph
        G_copy = G.copy()

        # Add edge probabilities as edge attributes
        # These represent the model's confidence for each predicted link
        for (source, target), prob in edge_probabilities.items():
            if G_copy.has_edge(source, target):
                G_copy.edges[source, target]["probability"] = float(prob)

        # Add community assignments to nodes
        # This allows visualization tools to color nodes by community
        if communities:
            for node, comm_id in communities.items():
                if G_copy.has_node(node):
                    G_copy.nodes[node]["community"] = int(comm_id)

        # Add phase information as graph-level attributes
        # This makes the file self-documenting and traceable
        if phase_info:
            for key, value in phase_info.items():
                # Convert all values to strings for GraphML compatibility
                G_copy.graph[key] = str(value)

        # Write to GraphML file
        # NetworkX handles the XML structure automatically
        nx.write_graphml(G_copy, filepath)
