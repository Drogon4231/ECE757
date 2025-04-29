import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# ----------------------------
# 1. Load & Prepare Data
# ----------------------------
df = pd.read_csv("/Users/harshithkantamneni/Desktop/757_Project/code/ECE757/src/benchmark_results 1.csv")


# Feature Engineering
df["nodes_per_partition"] = df["num_nodes"] / df["partition_size"]
df["edge_density"] = df["num_edges"] / (df["num_nodes"]**2)
df["size_ratio"] = df["matrix_size"] / df["partition_size"]

# Features
X = df[["num_nodes", "num_edges", "average_degree", "nodes_per_partition", "edge_density", "size_ratio"]].values

# Target: runtime (min-max scaled to [0, 1])
runtime_scaler = MinMaxScaler()
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)
y = runtime_scaler.fit_transform(df[["runtime"]]).flatten()

# ----------------------------
# 2. Train/Val/Test Split
# ----------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# ----------------------------
# 3. Device Configuration (GPU or CPU)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ----------------------------
# 4. Define Neural Network Model
# ----------------------------
class ConfigPredictor(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ConfigPredictor().to(device)

# ----------------------------
# 5. Training Setup
# ----------------------------
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

# ----------------------------
# 6. Training Loop
# ----------------------------
EPOCHS = 150
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t.to(device))
    loss = loss_fn(outputs, y_train_t.to(device))
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t.to(device))
        val_loss = loss_fn(val_outputs, y_val_t.to(device))
        scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_config_predictor.pt")

# ----------------------------
# 7. Save TorchScript Model and Scalers
# ----------------------------
model.load_state_dict(torch.load("best_config_predictor.pt"))
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("config_predictor.pt")
print("✅ Saved TorchScript model.")

# Save scalers
joblib.dump(runtime_scaler, "runtime_scaler.pkl")
joblib.dump(feature_scaler, "feature_scaler.pkl")
print("✅ Saved model and scalers")