import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# Config
# ----------------------------
partition_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data_path = "data/ml_data/training_data.csv"
model_path = "scripts/models/model.pkl"
scaler_path = "scripts/models/scaler.pkl"

# ----------------------------
# Load model and scaler
# ----------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ----------------------------
# Load benchmark/training data
# ----------------------------
df = pd.read_csv(data_path)

# ----------------------------
# Evaluation
# ----------------------------
print("matrix_size | true_best_partition | predicted_best_partition | runtimes (partition: predicted_runtime)")

results = []

for matrix_size in sorted(df['matrix_size'].unique()):
    subset = df[df['matrix_size'] == matrix_size]

    # True best
    true_best_partition = subset.loc[subset['runtime'].idxmin(), 'partition_size']

    # Predict
    features = []
    for p in partition_sizes:
        size_ratio = matrix_size / p
        log_matrix_size = np.log1p(matrix_size)
        log_partition_size = np.log1p(p)
        features.append([matrix_size, p, size_ratio, log_matrix_size, log_partition_size])

    feature_cols = ['matrix_size', 'partition_size', 'size_ratio', 'log_matrix_size', 'log_partition_size']
    features_df = pd.DataFrame(features, columns=feature_cols)

    # Scale and predict
    scaled = scaler.transform(features_df)
    preds = model.predict(scaled)

    predicted_best_idx = np.argmin(preds)
    predicted_best_partition = partition_sizes[predicted_best_idx]

    runtimes_str = ", ".join(f"{p}:{int(rt)}" for p, rt in zip(partition_sizes, preds))

    results.append({
        "matrix_size": matrix_size,
        "true_best_partition": true_best_partition,
        "predicted_best_partition": predicted_best_partition,
        "runtimes_str": runtimes_str
    })

    print(f"{matrix_size:11} | {true_best_partition:18} | {predicted_best_partition:23} | {runtimes_str}")
    print("-" * 100)

# ----------------------------
# Save results
# ----------------------------
os.makedirs("results", exist_ok=True)
out_df = pd.DataFrame(results)
out_df.to_csv("results/diagnostics_summary.csv", index=False)
print("📄 Saved diagnostic results to results/diagnostics_summary.csv")