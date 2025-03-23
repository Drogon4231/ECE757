# 2. Feature Extraction from Circuit Graphs
import networkx as nx

def extract_graph_features(circuit_file):
    G = nx.Graph()
    with open(circuit_file, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'density': nx.density(G)
    }
