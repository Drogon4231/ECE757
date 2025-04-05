import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 1. Load and preprocess data
df = pd.read_csv("tdg_features_output.csv")

# Drop completely empty or irrelevant columns
df = df.dropna(axis=1, how='all')

# Drop string columns (e.g. filenames) that are not useful for training
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

# Check for NaNs and fill or drop
df = df.dropna()

# Separate features and target
X = df.drop(columns=["matrix_size"])
y = df["matrix_size"]

# Encode matrix size (e.g. 32, 64, 128...) as classification labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for use during inference
joblib.dump(label_encoder, "matrix_size_label_encoder.joblib")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "tnn_feature_scaler.joblib")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. Define Tiny Neural Net
class TinyNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TinyNeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 3. Initialize model
input_size = X_train_tensor.shape[1]
num_classes = len(np.unique(y_encoded))
model = TinyNeuralNet(input_size, num_classes)

# 4. Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted == y_test_tensor).float().mean().item()

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 6. Save model
torch.save(model.state_dict(), "tiny_model.pt")