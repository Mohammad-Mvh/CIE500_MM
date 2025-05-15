# %%
from torch_geometric.utils import dense_to_sparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import swmmio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch, DataLoader


# %%
# Load SWMM model
model = swmmio.Model("BellingeSWMM_5min.inp")
# %%
# Extract nodes
storage_nodes = model.inp.storage.index.tolist(
) if hasattr(model.inp, 'storage') else []
junctions = model.inp.junctions.index.tolist(
) if hasattr(model.inp, 'junctions') else []
outfalls = model.inp.outfalls.index.tolist(
) if hasattr(model.inp, 'outfalls') else []
nodes = junctions + outfalls + storage_nodes

# Extract node coordinates
node_positions = model.inp.coordinates[["X", "Y"]].to_dict(orient="index")

# Create directed graph
G = nx.DiGraph()

# Extract links
all_links = model.inp.conduits.index.tolist()
if hasattr(model.inp, "pumps"):
    all_links += model.inp.pumps.index.tolist()
if hasattr(model.inp, "orifices"):
    all_links += model.inp.orifices.index.tolist()
if hasattr(model.inp, "weirs"):
    all_links += model.inp.weirs.index.tolist()

for link in all_links:
    start, end = None, None

    if link in model.inp.conduits.index:
        start, end = model.inp.conduits.loc[link, ["InletNode", "OutletNode"]]
    elif hasattr(model.inp, "pumps") and link in model.inp.pumps.index:
        start, end = model.inp.pumps.loc[link, ["InletNode", "OutletNode"]]
    elif hasattr(model.inp, "orifices") and link in model.inp.orifices.index:
        start, end = model.inp.orifices.loc[link, ["InletNode", "OutletNode"]]
    elif hasattr(model.inp, "weirs") and link in model.inp.weirs.index:
        start, end = model.inp.weirs.loc[link, ["InletNode", "OutletNode"]]

    if start in node_positions and end in node_positions:
        G.add_edge(start, end, link=link)

# %%
# Node features: Elevation and MaxDepth
node_features = model.inp.junctions.copy()
node_features["InvertElev"] = node_features["InvertElev"].astype(float)
node_features["MaxDepth"] = node_features["MaxDepth"].astype(float)
node_features = node_features[["InvertElev", "MaxDepth"]]

# Add outfalls: Assume MaxDepth=0
outfalls_df = model.inp.outfalls.copy()
outfalls_df["InvertElev"] = outfalls_df["InvertElev"].astype(float)
outfalls_df["MaxDepth"] = 0.0  # Assume 0 for outfalls
outfalls_df = outfalls_df[["InvertElev", "MaxDepth"]]
node_features = pd.concat([node_features, outfalls_df])

# Add storage (already have both MaxDepth and InvertElev)
storage_df = model.inp.storage.copy()
storage_df = storage_df.rename(
    columns={"MaxD": "MaxDepth"})  # Rename MaxD to MaxDepth
storage_df["InvertElev"] = storage_df["InvertElev"].astype(float)
storage_df["MaxDepth"] = storage_df["MaxDepth"].astype(float)
storage_df = storage_df[["InvertElev", "MaxDepth"]]
node_features = pd.concat([node_features, storage_df])


# Add coordinates to all nodes as features
for node_id, coords in node_positions.items():
    if node_id in node_features.index:
        node_features.at[node_id, "X"] = coords["X"]
        node_features.at[node_id, "Y"] = coords["Y"]
    else:
        node_features.loc[node_id] = [0.0, 0.0, coords["X"], coords["Y"]]

# %%
# Find the distance of each node from the shore line
x1, y1 = 583096, 6130674  # Example shoreline point 1
x2, y2 = 587461, 6135039  # Example shoreline point 2
shore_vec = np.array([x2 - x1, y2 - y1])
shore_norm = np.linalg.norm(shore_vec)

# Calculate distance


def point_to_line_distance(x, y):
    point_vec = np.array([x - x1, y - y1])
    proj_length = np.dot(point_vec, shore_vec) / shore_norm
    proj_point = np.array([x1, y1]) + (proj_length / shore_norm) * shore_vec
    return np.linalg.norm(np.array([x, y]) - proj_point)


node_coords = np.stack(
    [node_features["X"].values, node_features["Y"].values], axis=1)
distances = np.array([point_to_line_distance(x, y)
                      for x, y in node_coords])
node_features["Distance"] = distances

# Normalize distance to be between 0.1 and 0.9
node_features["Distance"] = (node_features["Distance"] - node_features["Distance"].min()) / \
    (node_features["Distance"].max() - node_features["Distance"].min())
node_features["Distance"] = 0.1 + 0.8 * node_features["Distance"]

# %%
# Link features: Length, Roughness, From Node, and To Node
link_features = model.inp.conduits.copy()
link_features["Length"] = link_features["Length"].astype(float)
link_features["Roughness"] = link_features["Roughness"].astype(float)
link_features["InletNode"] = link_features["InletNode"].astype(str)
link_features["OutletNode"] = link_features["OutletNode"].astype(str)
link_features["CrestHeight"] = 0.0  # Assume 0 for conduits
link_features = link_features[[
    "Length", "Roughness", "InletNode", "OutletNode", "CrestHeight"]]

# Add pumps (all features = 0.0 except InletNode and OutletNode)
pumps_df = model.inp.pumps.copy()
pumps_df["Length"] = 0.0
pumps_df["Roughness"] = 0.0
pumps_df["InletNode"] = pumps_df["InletNode"].astype(str)
pumps_df["OutletNode"] = pumps_df["OutletNode"].astype(str)
pumps_df["CrestHeight"] = 0.0
pumps_df = pumps_df[["Length", "Roughness",
                     "InletNode", "OutletNode", "CrestHeight"]]

# Add orifices (all features = 0.0 except InletNode and OutletNode)
orifices_df = model.inp.orifices.copy()
orifices_df["Length"] = 0.0
orifices_df["Roughness"] = 0.0
orifices_df["InletNode"] = orifices_df["InletNode"].astype(str)
orifices_df["OutletNode"] = orifices_df["OutletNode"].astype(str)
orifices_df["CrestHeight"] = 0.0
orifices_df = orifices_df[["Length", "Roughness",
                           "InletNode", "OutletNode", "CrestHeight"]]

# add weirs
weirs_df = model.inp.weirs.copy()
weirs_df["Length"] = 0.0  # Assume 0 for weirs
weirs_df["Roughness"] = 0.0  # Assume 0 for weirs
weirs_df["InletNode"] = weirs_df["InletNode"].astype(str)
weirs_df["OutletNode"] = weirs_df["OutletNode"].astype(str)
weirs_df["CrestHeight"] = weirs_df["CrestHeight"].astype(float)
weirs_df = weirs_df[["Length", "Roughness",
                     "InletNode", "OutletNode", "CrestHeight"]]
link_features = pd.concat([link_features, weirs_df])


# %%
# Assign node features into the graph
for node_id, attrs in node_features.iterrows():
    if node_id in G.nodes:
        G.nodes[node_id]["elev"] = attrs["InvertElev"]
        G.nodes[node_id]["depth"] = attrs["MaxDepth"]
        G.nodes[node_id]["X"] = attrs["X"]
        G.nodes[node_id]["Y"] = attrs["Y"]
        G.nodes[node_id]["Distance"] = attrs["Distance"]

# Assign link attributes into the graph
for link_id, attrs in link_features.iterrows():
    for u, v, data in G.edges(data=True):
        if data.get("link") == link_id:
            data["length"] = attrs["Length"]
            data["roughness"] = attrs["Roughness"]

# %%
# Remove edges with missing features (e.g., no 'length' or 'roughness')
edges_to_remove = []

for u, v, data in G.edges(data=True):
    if ("length" not in data) or ("roughness" not in data):
        edges_to_remove.append((u, v))

G.remove_edges_from(edges_to_remove)

print(f"Removed {len(edges_to_remove)} edges with missing features.")


# %%
# Define parameters for the GCN
lookback = 24  # Previous time steps for rainfall input
prediction_horizon = 1
train_scenarios = [f"sc{str(i+1).zfill(2)}" for i in range(54)]
test_scenarios = [f"sc{str(i+1).zfill(2)}" for i in range(54, 74)]
outfall_nodes = ['F74F360', 'G60K220', 'G60U02F', 'G70U01F',
                 'G71U02F', 'G71U10F', 'G72U07R', 'G75U03F', 'G80U03F']
# %%
# Read rainfall data
rain_df = pd.read_csv("RainfallTimeSeries.csv")
rain_df.drop(columns=["Time"], inplace=True)

# Normalize rainfall
rain_scaler = MinMaxScaler()
rain_norm = pd.DataFrame(rain_scaler.fit_transform(
    rain_df), columns=rain_df.columns)
# %%
# Read and normalize flow data
flow_scalers = {}
flow_data = {}

for node in outfall_nodes:
    df = pd.read_excel("OutfallTimeSeries.xlsx", sheet_name=node)
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    flow_scalers[node] = scaler
    flow_data[node] = {col: norm_df[col].values for col in norm_df.columns}

# %%
# Create adjacency matrix
node_list = list(G.nodes())
node_index = {n: i for i, n in enumerate(node_list)}
A = nx.to_numpy_array(G, nodelist=node_list)
A = A + np.eye(len(A))  # Add self-connections
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(1)))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt
A_hat = torch.tensor(A_hat, dtype=torch.float32)
# Convert adjacency matrix A_hat to edge_index for PyTorch
edge_index, edge_weight = dense_to_sparse(A_hat)
# %%
# Create node feature matrix
node_features_df = pd.DataFrame({
    "elev": [G.nodes[n].get("elev", 0.0) for n in node_list],
    "depth": [G.nodes[n].get("depth", 0.0) for n in node_list],
    "X": [G.nodes[n].get("X", 0.0) for n in node_list],
    "Y": [G.nodes[n].get("Y", 0.0) for n in node_list],
    "Distance": [G.nodes[n].get("Distance", 0.0) for n in node_list]
}, index=pd.Index(node_list, name="node_id"))  # Ensure index is 1D and labeled

# Normalize node features
node_scaler = MinMaxScaler()
node_features_norm = torch.tensor(
    node_scaler.fit_transform(node_features_df.values),
    dtype=torch.float32
)

# Store distance coefficients separately for use with rainfall
distance_coeffs = torch.tensor(
    node_features_df["Distance"].values, dtype=torch.float32
)  # shape should be (N,)

# %%
# Constants of the model
num_nodes = len(node_list)
num_outfalls = len(outfall_nodes)
outfall_indices = [node_index[n] for n in outfall_nodes]

# %%
# Create training and test sets


def create_graph_samples(scenario_ids):
    dataset = []
    for sid in scenario_ids:
        rain_series = rain_norm[sid].values  # shape: (T,)
        flow_series = [flow_data[n][sid] for n in outfall_nodes]
        flow_series = np.stack(flow_series, axis=1)  # shape: (T, 9)

        T = len(rain_series)
        for t in range(lookback, T - prediction_horizon):
            # Rainfall (lookback window)
            past_rain = rain_series[t - lookback:t]  # shape: (lookback,)
            past_rain_scaled = torch.stack([
                torch.tensor(past_rain[i] *
                             distance_coeffs, dtype=torch.float32)
                for i in range(lookback)
            ], dim=1)  # shape: (N, lookback)

            # Time encoding
            time_feat = torch.tensor(
                [t / T], dtype=torch.float32).repeat(num_nodes, 1)

            # Flow history (last 3 time steps)
            flow_hist = torch.zeros((num_nodes, 3), dtype=torch.float32)
            for i, node in enumerate(outfall_nodes):
                node_idx = node_index[node]
                flow_hist[node_idx] = torch.tensor(
                    flow_series[t - 3:t, i], dtype=torch.float32)

            # Concatenate all features
            node_input = torch.cat([
                past_rain_scaled,
                # flow_hist,
                node_features_norm,
                time_feat
            ], dim=1)

            # Target is only the outfall flows (shape: [num_outfalls])
            y = torch.tensor(flow_series[t], dtype=torch.float32)

            data = Data(
                x=node_input,
                edge_index=edge_index,
                y=y,
                outfall_indices=torch.tensor(
                    outfall_indices, dtype=torch.long),
                scenario_id=int(sid.replace('sc', ''))
            )
            dataset.append(data)
    return dataset


# %%
# Create datasets
train_dataset = create_graph_samples(train_scenarios)
test_dataset = create_graph_samples(test_scenarios)
# Inspect samples to make sure structure is correct
for i in range(3):
    sample = train_dataset[i]
    print(f"\nSample {i}")
    print("Input x shape:", sample.x.shape)
    # show first 5 nodes' features
    print("First few rows of x:\n", sample.x[:5])
    print("Output y shape:", sample.y.shape)
    print("y (target outfall discharges):", sample.y)
    print("Outfall indices:", sample.outfall_indices)
    print("Scenario ID:", sample.scenario_id)
# %%
# Define the GCN model


class GCNFlowPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, 48)
        self.gcn2 = GCNConv(48, 48)
        # self.gcn3 = GCNConv(64, 48)
        self.linear = nn.Linear(48, out_channels)

    def forward(self, x, edge_index, outfall_indices):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        # x = self.gcn3(x, edge_index)
        # x = F.relu(x)
        outfall_x = x[outfall_indices]         # shape: [9, hidden]
        out = self.linear(outfall_x)           # shape: [9, 1]
        out = torch.sigmoid(out).squeeze(-1)   # shape: [9]
        return out


# %%
# Training Setup
# Initialize model

input_dim = lookback + node_features_norm.shape[1] + 1

model = GCNFlowPredictor(
    in_channels=input_dim,
    out_channels=1
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.MSELoss()


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.75, verbose=True
)

# %%
# Training Loop
model.train()
epoch_losses = []  # Store average loss per epoch

for epoch in range(1):
    total_loss = 0
    sample_count = 0

    # Process samples one at a time
    for sample in train_dataset:  # Directly iterate through dataset
        optimizer.zero_grad()

        # Forward pass
        out = model(sample.x, sample.edge_index, sample.outfall_indices)
        loss = loss_fn(out, sample.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        sample_count += 1

    avg_loss = total_loss / sample_count
    epoch_losses.append(avg_loss)

    # Update scheduler
    print(
        f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    scheduler.step(avg_loss)

# %%
# Plot training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(epoch_losses)+1), epoch_losses,
         label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# %%
# Testing
model.eval()
outfall_results = {node_id: {'pred': [], 'true': []}
                   for node_id in outfall_indices}

with torch.no_grad():
    for sample in test_dataset:
        out = model(sample.x, sample.edge_index, sample.outfall_indices)
        pred = out.cpu().numpy()
        true = sample.y.cpu().numpy()

        for i, node_id in enumerate(outfall_indices):
            outfall_results[node_id]['pred'].append(pred[i])
            outfall_results[node_id]['true'].append(true[i])

# %%
# Calculate and print MSE for each outfall
print("Outfall MSE Results:")
for node_id in outfall_indices:
    pred = torch.tensor(outfall_results[node_id]['pred'])
    true = torch.tensor(outfall_results[node_id]['true'])
    mse = F.mse_loss(pred, true)
    print(f"Outfall {node_id}: MSE = {mse.item():.6f}")
# Print outfall name correspondint to its node_id

# %%
# Compute average discharge across test scenarios for each outfall
num_test_scenarios = len(test_scenarios)
num_samples = len(outfall_results[outfall_indices[0]]['pred'])
timesteps_per_scenario = num_samples // num_test_scenarios

# Reshape and average per time step, then denormalize
avg_results = {}
for i, node_id in enumerate(outfall_indices):
    node_name = outfall_nodes[i]
    scaler = flow_scalers[node_name]

    pred_all = np.array(outfall_results[node_id]['pred'])
    true_all = np.array(outfall_results[node_id]['true'])

    # Reshape to (num_test_scenarios, timesteps_per_scenario)
    pred_matrix = pred_all.reshape(num_test_scenarios, timesteps_per_scenario)
    true_matrix = true_all.reshape(num_test_scenarios, timesteps_per_scenario)

    # Average across test scenarios
    pred_avg = pred_matrix.mean(axis=0)
    true_avg = true_matrix.mean(axis=0)

    # Denormalize to show meaningful values
    try:
        if hasattr(scaler, 'mean_'):  # StandardScaler
            pred_denorm = (pred_avg * scaler.scale_[0]) + scaler.mean_[0]
            true_denorm = (true_avg * scaler.scale_[0]) + scaler.mean_[0]
        else:
            pred_denorm = pred_avg * \
                (scaler.data_max_[0] - scaler.data_min_[0]) + \
                scaler.data_min_[0]
            true_denorm = true_avg * \
                (scaler.data_max_[0] - scaler.data_min_[0]) + \
                scaler.data_min_[0]

        avg_results[node_id] = {
            'pred': pred_denorm,
            'true': true_denorm
        }

    except Exception as e:
        print(f"Error denormalizing {node_name}: {str(e)}")
        avg_results[node_id] = {
            'pred': pred_avg,
            'true': true_avg
        }

# Plotting
num_outfalls = len(outfall_indices)
fig, axes = plt.subplots(num_outfalls, 1, figsize=(
    10, 3*num_outfalls), sharex=True)

for i, node_id in enumerate(outfall_indices):
    data = avg_results[node_id]

    plt.figure(figsize=(10, 3))
    plt.plot(data['true'], label='True', color='black', linewidth=1.5)
    plt.plot(data['pred'], label='Predicted', color='blue', linestyle='--')

    plt.title(f"Outfall {outfall_nodes[i]}")
    plt.ylabel("Discharge (m³/s)")
    plt.xlabel("Time Step (5-min intervals)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
