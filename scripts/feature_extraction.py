# 2. Feature Extraction from Circuit Graphs
import os
import re
import networkx as nx
import pandas as pd

# --- Step 1: Parse runtime data from the .hpp file ---
def parse_runtime_data_fixed(runtime_text):
    pattern = re.compile(r'const std::vector<int>\s+matrix(\d+)_rt\s*=\s*{([^}]*)}')
    matrix_data = {}

    for match in pattern.finditer(runtime_text):
        matrix_size = int(match.group(1))
        raw_values = match.group(2).split(',')
        runtimes = []
        for val in raw_values:
            val = val.strip()
            if val.isdigit():
                runtimes.append(int(val))
        matrix_data[matrix_size] = runtimes

    return matrix_data

# --- Step 2: Parse TDG graph file (.txt) ---
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
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'density': nx.density(G)
    }

# --- Step 3: Load and process everything ---
def extract_all_features(runtime_file_path, tdg_directory_path, output_csv_path):
    # Load runtime file
    with open(runtime_file_path, 'r') as f:
        runtime_text = f.read()

    runtime_vectors = parse_runtime_data_fixed(runtime_text)
    
    # Gather all TDG .txt files in the directory
    tdg_files = [f for f in os.listdir(tdg_directory_path) if f.endswith(".txt")]
    
    # Aggregate features
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
                    'runtime': avg_runtime
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Features saved to: {output_csv_path}")

# Example usage
extract_all_features(
    runtime_file_path="/mnt/data/runtime_data.hpp",
    tdg_directory_path="/mnt/data",
    output_csv_path="/mnt/data/all_tdg_features_output.csv"
)