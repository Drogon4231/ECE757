import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os

# Load and prepare data
df = pd.read_csv("data/ml_data/training_data.csv")

# Enhanced feature engineering
df["size_ratio"] = df["matrix_size"] / df["partition_size"]
df["log_matrix_size"] = np.log1p(df["matrix_size"])
df["log_partition_size"] = np.log1p(df["partition_size"])
df["abs_diff"] = np.abs(df["matrix_size"] - df["partition_size"])
df["modulo"] = df["matrix_size"] % df["partition_size"]
df["ratio"] = np.where(
    df["matrix_size"] > 0,
    df["partition_size"] / df["matrix_size"],
    0
)
df["log_abs_diff"] = np.log1p(df["abs_diff"])
df["interaction"] = df["matrix_size"] * df["partition_size"]
df["squared_matrix_size"] = df["matrix_size"] ** 2
df["squared_partition_size"] = df["partition_size"] ** 2
df["size_ratio_squared"] = (df["matrix_size"] / df["partition_size"])**2
df["cubic_matrix_size"] = df["matrix_size"]**3
df["size_product"] = df["matrix_size"] * df["partition_size"]
df["size_diff_ratio"] = df["abs_diff"] / (df["matrix_size"] + 1e-6)

# Global constants
PARTITION_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DEFAULT_GRAPH_FEATURES = [42438, 53558, 1.26203]

features = [
    "matrix_size", "partition_size", "size_ratio", "log_matrix_size",
    "log_partition_size", "abs_diff", "modulo", "ratio", "log_abs_diff",
    "interaction", "squared_matrix_size", "squared_partition_size",
    "size_ratio_squared", "cubic_matrix_size", "size_product", "size_diff_ratio",
    "num_nodes", "num_edges", "average_degree"
]

# Data validation
print("Missing values per column:")
print(df[features].isnull().sum())

# Remove extreme outliers (top 1%)
df = df[df["runtime"] < df["runtime"].quantile(0.99)]

X = df[features].values
y = np.log1p(df["runtime"].values)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [6, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1]
}

grid = GridSearchCV(
    XGBRegressor(
        n_estimators=300,
        tree_method="hist",
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"\nBest parameters: {grid.best_params_}")

# Final model
xgb_model = grid.best_estimator_

# Cross-validation
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
xgb_rmse = np.sqrt(-xgb_scores.mean())
print(f"XGBoost CV RMSE: {xgb_rmse:.4f}")

# Train and evaluate
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost Test RMSE: {test_rmse:.4f}")

# Ensure the output directory exists
os.makedirs("scripts/models", exist_ok=True)

# Dump scaler
joblib.dump(scaler, "scripts/models/scaler.pkl")

# Random Forest baseline
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
print(f"Random Forest CV RMSE: {np.sqrt(-rf_scores.mean()):.4f}")

# Save model
joblib.dump(xgb, "scripts/models/model.pkl")
print("Model saved as 'model.pkl'")

# Feature importance
plt.figure(figsize=(12,8))
sorted_idx = xgb_model.feature_importances_.argsort()
plt.barh(np.array(features)[sorted_idx], xgb_model.feature_importances_[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=300)
plt.close()

# Enhanced prediction function
def predict_best_partition_xgb(matrix_size, model, scaler, 
                             partition_sizes=PARTITION_SIZES,
                             graph_features=DEFAULT_GRAPH_FEATURES):
    feats = []
    num_nodes, num_edges, avg_degree = graph_features
    
    for p in partition_sizes:
        feats.append([
            matrix_size, p, matrix_size/p,
            np.log1p(matrix_size), np.log1p(p),
            abs(matrix_size - p), matrix_size % p,
            p/matrix_size if matrix_size > 0 else 0,
            np.log1p(abs(matrix_size - p)),
            matrix_size * p, matrix_size**2, p**2,
            (matrix_size/p)**2 if p > 0 else 0,
            matrix_size**3,
            matrix_size * p,  # size_product
            abs(matrix_size - p)/(matrix_size + 1e-6),  # size_diff_ratio
            num_nodes, num_edges, avg_degree
        ])
    
    feats_scaled = scaler.transform(feats)
    preds = np.expm1(model.predict(feats_scaled))
    best_idx = np.argmin(preds)
    return partition_sizes[best_idx], preds[best_idx], preds

# Example usage
matrix_size = 32
best_p, best_runtime, all_preds = predict_best_partition_xgb(matrix_size, xgb_model, scaler)
print(f"\nFor matrix_size={matrix_size}:")
print(f"Predicted best partition: {best_p}")
print(f"Predicted runtime: {best_runtime:.2f}")

# Diagnostic checks
print("\nActual runtimes for predicted partition:")
print(df[(df["matrix_size"] == 32) & (df["partition_size"] == best_p)]["runtime"].describe())

# Error analysis
df["pred_runtime"] = np.expm1(xgb_model.predict(X_scaled))
df["error"] = df["runtime"] - df["pred_runtime"]
df["abs_error"] = np.abs(df["error"])

print("\nError analysis:")
print(df[["matrix_size", "partition_size", "runtime", "pred_runtime", "error"]].describe())

# Worst predictions
print("\nTop 10 Worst Predictions:")
print(df.nlargest(10, "abs_error")[["matrix_size", "partition_size", "runtime", "pred_runtime", "error"]])

# Error visualization
plt.figure(figsize=(12,6))
plt.scatter(df["runtime"], df["error"], alpha=0.3)
plt.axhline(0, color="r", linestyle="--")
plt.xscale("log")
plt.xlabel("True Runtime (log scale)")
plt.ylabel("Prediction Error")
plt.title("Error Distribution")
plt.savefig("error_distribution_log.png", dpi=300)
plt.close()

# Learning curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    xgb_model, X_scaled, y, cv=5,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, np.sqrt(-train_scores.mean(1)), "o-", label="Training")
plt.plot(train_sizes, np.sqrt(-test_scores.mean(1)), "o-", label="Validation")
plt.xlabel("Training Examples")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("learning_curve.png", dpi=300)
plt.close()

# Prediction vs actual plot
matrix_size = 32
if matrix_size in df["matrix_size"].unique():
    true_runtimes = (
        df[df["matrix_size"] == matrix_size]
        .groupby("partition_size")["runtime"]
        .mean()
        .reindex(PARTITION_SIZES)
    )
    
    _, _, pred_runtimes = predict_best_partition_xgb(matrix_size, xgb_model, scaler)
    
    plt.figure(figsize=(14,8))
    plt.plot(PARTITION_SIZES, true_runtimes, "o-", label="True Runtime")
    plt.plot(PARTITION_SIZES, pred_runtimes, "x--", label="XGBoost Predicted")
    
    best_idx = np.argmin(pred_runtimes)
    plt.annotate(
        f"Best: {PARTITION_SIZES[best_idx]}",
        (PARTITION_SIZES[best_idx], pred_runtimes[best_idx]),
        textcoords="offset points",
        xytext=(0,10),
        ha="center"
    )
    
    plt.title(f"Matrix Size = {matrix_size}")
    plt.xlabel("Partition Size")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"matrix_{matrix_size}_xgb_prediction.png", dpi=300)
    plt.close()
    
    comparison = pd.DataFrame({
        "Partition": PARTITION_SIZES,
        "True Runtime": true_runtimes,
        "Predicted Runtime": pred_runtimes,
        "Absolute Error": np.abs(true_runtimes - pred_runtimes)
    })
    print("\nPrediction vs Actual:")
    print(comparison.dropna().round(2))
else:
    print(f"\nMatrix size {matrix_size} not found in data")

# Runtime distribution
plt.figure(figsize=(12,6))
plt.hist(df["runtime"], bins=50, log=True)
plt.title("Runtime Distribution (Log Scale)")
plt.xlabel("Runtime (seconds)")
plt.ylabel("Frequency")
plt.savefig("runtime_distribution.png", dpi=300)
plt.close()


df["pred_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
