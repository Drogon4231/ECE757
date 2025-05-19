import os
import re
import networkx as nx
import pandas as pd

# --- Step 1: Parse runtime data from .hpp ---
def parse_runtime_data_fixed(runtime_text):
    pattern = re.compile(r'const std::vector<int>\s+matrix(\d+)_rt\s*=\s*{([^}]*)}')
    matrix_data = {}

    for match in pattern.finditer(runtime_text):
        matrix_size = int(match.group(1))
        raw_values = match.group(2).split(',')
        runtimes = [int(val.strip()) for val in raw_values if val.strip().isdigit()]
        matrix_data[matrix_size] = runtimes

    return matrix_data

# --- Step 2: Parse TDG .txt file ---
def parse_tdg_file_from_path(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    num_nodes = int(lines[0].strip())
    G = nx.DiGraph()

    for line in lines[1:]:
        parts = re.findall(r'"(\d+)"', line)
        if not parts:
            continue
        node = int(parts[0])
        G.add_node(node)
        for dep in parts[1:]:
            G.add_edge(int(dep), node)

    return {
        'filename': os.path.basename(path),
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() else 0,
        'density': nx.density(G)
    }

# --- Step 3: Extract all features and runtimes ---
def extract_all_features(runtime_file_path, tdg_directory_path, output_csv_path):
    # Load and parse runtime data
    with open(runtime_file_path, 'r') as f:
        runtime_text = f.read()

    runtime_vectors = parse_runtime_data_fixed(runtime_text)
    
    # Get all TDG .txt files
    tdg_files = [f for f in os.listdir(tdg_directory_path) if f.endswith(".txt")]
    
    # Aggregate into one dataset
    rows = []
    for tdg_file in tdg_files:
        tdg_path = os.path.join(tdg_directory_path, tdg_file)
        tdg_features = parse_tdg_file_from_path(tdg_path)

        for matrix_size, runtimes in runtime_vectors.items():
            num_partitions = len(runtimes) // 4
            for i in range(num_partitions):
                avg_runtime = sum(runtimes[i*4:(i+1)*4]) / 4
                row = {
                    **tdg_features,
                    'matrix_size': matrix_size,
                    'partition_size': (i + 1) * 10,  # Assumes partitions are 10, 20, ..., 100
                    'runtime': avg_runtime
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Features saved to: {output_csv_path}")

# --- Run the extractor ---
if __name__ == "__main__":
    extract_all_features(
        runtime_file_path="data/ml_data/runtime_data.hpp",
        tdg_directory_path="data/graphs",
        output_csv_path="data/ml_data/training_data.csv"
    )