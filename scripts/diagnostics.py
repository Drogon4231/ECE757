import pandas as pd
import numpy as np
import torch
import joblib

# Load model and scalers
model = torch.jit.load("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/runtime_regressor.pt", map_location="cpu")
model.eval()
feature_scaler = joblib.load("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/feature_scaler.pkl")
target_scaler = joblib.load("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/target_scaler.pkl")

# All possible partition sizes
partition_sizes = [10,20,30,40,50,60,70,80,90,100]

# Load data for true best comparison
df = pd.read_csv("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/benchmark_results 1.csv")

print("matrix_size | true_best_partition | predicted_best_partition | runtimes (partition: predicted_runtime)")

for matrix_size in sorted(df['matrix_size'].unique()):
    # True best partition size
    subset = df[df['matrix_size'] == matrix_size]
    true_best_partition = subset.loc[subset['runtime'].idxmin(), 'partition_size']

    # Predict for all partition sizes
    features = []
    for p in partition_sizes:
        size_ratio = matrix_size / p
        log_matrix_size = np.log1p(matrix_size)
        log_partition_size = np.log1p(p)
        features.append([matrix_size, p, size_ratio, log_matrix_size, log_partition_size])
    features = np.array(features)
    features_scaled = feature_scaler.transform(features)
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(input_tensor).cpu().numpy()
    # Inverse transform: first MinMaxScaler, then expm1
    preds_log = target_scaler.inverse_transform(preds_scaled)
    preds = np.expm1(preds_log).flatten()
    predicted_best_idx = np.argmin(preds)
    predicted_best_partition = partition_sizes[predicted_best_idx]
    runtimes_str = ", ".join(f"{p}:{int(rt)}" for p, rt in zip(partition_sizes, preds))
    print(f"{matrix_size:11} | {true_best_partition:18} | {predicted_best_partition:23} | {runtimes_str}")
    print("-" * 100)
