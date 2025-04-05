# 2. Feature Extraction from Circuit Graphs
import networkx as nx

def extract_graph_features(circuit_file):
    G = nx.DiGraph()  # Directed graph since tasks depend on others

    with open(circuit_file, 'r') as f:
        lines = f.readlines()
        num_nodes = int(lines[0])  # First line is total number of nodes

        for line in lines[1:]:
            tokens = line.strip().split()
            node_id = int(tokens[0])
            deps = list(map(int, tokens[2:])) if len(tokens) > 2 else []
            for d in deps:
                G.add_edge(d, node_id)  # Add edge from dependency -> current node

    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'density': nx.density(G)
    }