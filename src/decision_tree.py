import torch
import torch.nn as nn
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tdg_features_output.csv")

# Features & Target Variable
X = df[["num_nodes", "num_edges", "avg_degree", "density", "matrix_size"]].values
y = df["runtime"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
dt_model = DecisionTreeRegressor(max_depth=5)
dt_model.fit(X_train, y_train)

# Convert Decision Tree to a PyTorch Model
class DecisionTreeModel(nn.Module):
    def __init__(self, tree):
        super(DecisionTreeModel, self).__init__()
        self.tree = tree  # Store trained sklearn tree

    def forward(self, x):
        x = x.cpu().numpy()  # Convert PyTorch tensor to numpy
        return torch.tensor(self.tree.predict(x), dtype=torch.float32).cuda()

# Create PyTorch Decision Tree
torch_model = DecisionTreeModel(dt_model)
scripted_model = torch.jit.script(torch_model)
scripted_model.save("decision_tree_model.pt")  # Save model for CUDA