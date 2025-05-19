import os
import joblib
import numpy as np
import networkx as nx
import subprocess
import re
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ----------------------------
# Load XGBoost model (no scaler needed - we'll recreate)
# ----------------------------
model = joblib.load("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/xgboost_model.pkl")

# ----------------------------
# Feature engineering with proper column order
# ----------------------------
FEATURE_ORDER = [
    "matrix_size", "partition_size", "size_ratio", "log_matrix_size",
    "log_partition_size", "abs_diff", "modulo", "ratio", "log_abs_diff",
    "interaction", "squared_matrix_size", "squared_partition_size",
    "size_ratio_squared", "cubic_matrix_size", "size_product", "size_diff_ratio",
    "num_nodes", "num_edges", "average_degree"
]

def create_features(matrix_size, partition_size, graph_features):
    features = {
        "matrix_size": matrix_size,
        "partition_size": partition_size,
        "size_ratio": matrix_size / partition_size,
        "log_matrix_size": np.log1p(matrix_size),
        "log_partition_size": np.log1p(partition_size),
        "abs_diff": abs(matrix_size - partition_size),
        "modulo": matrix_size % partition_size,
        "ratio": partition_size / matrix_size if matrix_size > 0 else 0,
        "log_abs_diff": np.log1p(abs(matrix_size - partition_size)),
        "interaction": matrix_size * partition_size,
        "squared_matrix_size": matrix_size ** 2,
        "squared_partition_size": partition_size ** 2,
        "size_ratio_squared": (matrix_size / partition_size)**2,
        "cubic_matrix_size": matrix_size**3,
        "size_product": matrix_size * partition_size,
        "size_diff_ratio": abs(matrix_size - partition_size) / (matrix_size + 1e-6),
        "num_nodes": graph_features["num_nodes"],
        "num_edges": graph_features["num_edges"],
        "average_degree": graph_features["average_degree"]
    }
    
    # Ensure consistent column order
    return pd.DataFrame([features])[FEATURE_ORDER]

# ----------------------------
# Scaling recreation
# ----------------------------
def scale_features(features_df):
    """Recreate MinMax scaling logic"""
    # Note: In production, you should save/load your actual scaler
    # This is a temporary workaround
    scaler = MinMaxScaler()
    
    # Fake fit (scalers need to be fit before transform)
    # In reality, you should use the original training data ranges
    # These are placeholder ranges - replace with your actual training data ranges!
    dummy_ranges = pd.DataFrame(columns=FEATURE_ORDER)
    for col in FEATURE_ORDER:
        if 'matrix_size' in col:
            dummy_ranges[col] = [0, 100]  # Update with your actual max matrix size
        elif 'partition_size' in col:
            dummy_ranges[col] = [10, 100]
        elif 'num_nodes' in col:
            dummy_ranges[col] = [0, 100000]
        elif 'num_edges' in col:
            dummy_ranges[col] = [0, 500000]
        else:
            dummy_ranges[col] = [0, 100]  # Default fallback
    
    scaler.fit(dummy_ranges)
    return scaler.transform(features_df)

# ----------------------------
# Predict best partition size
# ----------------------------
def predict_best_partition(tdg_file, matrix_sizes):
    graph_features = extract_graph_features(tdg_file)
    results = []

    for matrix_size in matrix_sizes:
        # Generate all candidate features
        features_list = []
        for p in valid_partitions:
            features = create_features(matrix_size, p, graph_features)
            features_list.append(features)
        
        full_features = pd.concat(features_list)
        
        # Scale features (using dummy scaler - replace with real scaler!)
        scaled_features = scale_features(full_features)
        
        # Make predictions
        preds = np.expm1(model.predict(scaled_features))
        
        # Find best partition
        best_idx = np.argmin(preds)
        best_partition = valid_partitions[best_idx]
        best_runtime = preds[best_idx]

        results.append((matrix_size, best_partition, best_runtime))
        print(f"\n📦 Matrix: {matrix_size} → 🧠 Best Partition: {best_partition} (Predicted Runtime: {best_runtime:.0f} sec)")

        launch_partitioned_kernel(tdg_file, matrix_size, best_partition)

    # Save results
    graph_name = os.path.basename(tdg_file).replace(".txt", "")
    csv_name = f"best_partition_{graph_name}.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_size", "best_partition", "predicted_runtime"])
        writer.writerows(results)
    print(f"\n📄 Saved results to {csv_name}")

# ----------------------------
# Helper functions (unchanged)
# ----------------------------
def extract_graph_features(circuit_file):
    G = nx.DiGraph()
    with open(circuit_file, 'r') as f:
        lines = f.readlines()
        num_nodes = int(lines[0].strip())

        edge_pattern = re.compile(r'"(\d+)"\s*->\s*"(\d+)"')
        for line in lines:
            match = edge_pattern.findall(line)
            for src, dst in match:
                G.add_edge(int(src), int(dst))

    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes else 0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": avg_degree
    }

def launch_partitioned_kernel(tdg_file, matrix_size, partition_size):
    cmd = f"./examples/final_project_test {tdg_file} {matrix_size} {partition_size}"
    print(f"\n🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True)

# ----------------------------
# Main execution (unchanged)
# ----------------------------
if __name__ == "__main__":
    valid_partitions = [10,20,30,40,50,60,70,80,90,100]
    base_dir = "/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src"
    tdg_files = [
        "des_perf.txt",
        "aes_core.txt",
        "ac97_ctrl.txt",
        "vga_lcd.txt",
        "wb_dma.txt"
    ]
    matrix_opts = [2, 4, 8, 16, 32, 64]

    for tdg_file in tdg_files:
        full_path = os.path.join(base_dir, tdg_file)
        print(f"\n🔍 Processing {tdg_file}")
        predict_best_partition(full_path, matrix_opts)
