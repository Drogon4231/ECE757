import os
import torch
import joblib
import numpy as np
import networkx as nx
import subprocess

# ----------------------------
# Load model and scalers
# ----------------------------
model = torch.load("config_predictor.pth", map_location=torch.device("cpu"))
model.eval()
runtime_scaler = joblib.load("runtime_scaler.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")

# ----------------------------
# Extract features from TDG file
# ----------------------------
def extract_tdg_features(circuit_file, matrix_size, partition_size):
    G = nx.DiGraph()

    with open(circuit_file, 'r') as f:
        lines = f.readlines()
        num_nodes = int(lines[0])
        for line in lines[1:]:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            node_id = int(tokens[0])
            deps = list(map(int, tokens[2:])) if len(tokens) > 2 else []
            for d in deps:
                G.add_edge(d, node_id)

    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes else 0
    nodes_per_partition = num_nodes / partition_size
    edge_density = num_edges / (num_nodes ** 2) if num_nodes > 0 else 0
    size_ratio = matrix_size / partition_size

    return [num_nodes, num_edges, avg_degree, nodes_per_partition, edge_density, size_ratio]

# ----------------------------
# Predict Best Config
# ----------------------------
def predict_best_config(tdg_file, matrix_sizes, partition_sizes):
    best_time = float('inf')
    best_config = None

    for m in matrix_sizes:
        for p in partition_sizes:
            feat = extract_tdg_features(tdg_file, m, p)
            feat_scaled = feature_scaler.transform([feat])
            with torch.no_grad():
                norm_pred = model(torch.tensor(feat_scaled, dtype=torch.float32)).numpy().flatten()
                pred_runtime = runtime_scaler.inverse_transform(norm_pred.reshape(-1, 1))[0][0]

            print(f"Tested: Matrix={m}, Partition={p} → Runtime ≈ {pred_runtime:.2f} μs")
            if pred_runtime < best_time:
                best_time = pred_runtime
                best_config = (m, p)

    return best_config, best_time

# ----------------------------
# Launch Kernel
# ----------------------------
def launch_partitioned_kernel(tdg_file, matrix_size, partition_size):
    cmd = f"./examples/final_project_test {tdg_file} {matrix_size} {partition_size}"
    print(f"\n🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True)

# ----------------------------
# Run End-to-End
# ----------------------------
if __name__ == "__main__":
    tdg_file = "./benchmark/des_perf.txt"  # <-- update this
    matrix_opts = [8, 16, 32]
    partition_opts = [2, 4, 8]

    best_cfg, best_pred_time = predict_best_config(tdg_file, matrix_opts, partition_opts)

    print(f"\n✅ Best Configuration:")
    print(f"Matrix Size: {best_cfg[0]}")
    print(f"Partition Size: {best_cfg[1]}")
    print(f"Predicted Runtime: {best_pred_time:.2f} μs")

    # Launch with both values
    launch_partitioned_kernel(tdg_file, best_cfg[0], best_cfg[1])