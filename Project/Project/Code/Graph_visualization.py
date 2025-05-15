# %%
from shapely.geometry import Point, LineString
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import swmmio
import pandas as pd

# %%
# Load the modified SWMM model
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
x1, y1 = 583096, 6130674  # Start of shoreline point 1
x2, y2 = 587461, 6135039  # End of shoreline point 2
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
# %% Print the number of nodes and edges
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# %%
# Draw the network
# Define node colors (green for outfalls, blue for others)
node_colors = ["red" if n in outfalls else "blue" for n in G.nodes]

# Use real-world coordinates for plotting
pos = {node: (coords["X"], coords["Y"])
       for node, coords in node_positions.items()}
plt.figure(figsize=(10, 10))
nx.draw(
    G,
    pos=pos,
    with_labels=False,
    node_color=node_colors,
    node_size=50,
    edge_color="green",
    width=1,
)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("SWMM Network Layout")
plt.show()
# %%
# Extract nodes and links as shapefile for QGIS
# Extract nodes as a GeoDataFrame
nodes_gdf = gpd.GeoDataFrame(
    columns=['node_id', 'elev', 'depth', 'X', 'Y', 'Distance', 'geometry'])

for node_id, data in G.nodes(data=True):
    if 'X' in data and 'Y' in data:
        point = Point(data['X'], data['Y'])
        nodes_gdf = pd.concat([nodes_gdf, gpd.GeoDataFrame({
            'node_id': [node_id],
            'elev': [data.get('elev', 0)],
            'depth': [data.get('depth', 0)],
            'X': [data['X']],
            'Y': [data['Y']],
            'Distance': [data.get('Distance', 0)],
            'geometry': [point]
        })], ignore_index=True)

# Set CRS (Coordinate Reference System) - adjust this to your actual CRS
nodes_gdf.crs = "EPSG:25832"

# Extract links as a GeoDataFrame
links_gdf = gpd.GeoDataFrame(
    columns=['link_id', 'from_node', 'to_node', 'length', 'roughness', 'geometry'])

for u, v, data in G.edges(data=True):
    if u in node_positions and v in node_positions:
        # Get coordinates for start and end nodes
        start_coords = node_positions[u]
        end_coords = node_positions[v]

        # Create LineString geometry
        line = LineString([(start_coords['X'], start_coords['Y']),
                           (end_coords['X'], end_coords['Y'])])

        links_gdf = pd.concat([links_gdf, gpd.GeoDataFrame({
            'link_id': [data.get('link', '')],
            'from_node': [u],
            'to_node': [v],
            'length': [data.get('length', 0)],
            'roughness': [data.get('roughness', 0)],
            'geometry': [line]
        })], ignore_index=True)

# Set CRS (should match nodes CRS)
links_gdf.crs = "EPSG:25832"

# Save to shapefiles
nodes_gdf.to_file("swmm_nodes.shp")
links_gdf.to_file("swmm_links.shp")

print(f"Saved {len(nodes_gdf)} nodes and {len(links_gdf)} links to shapefiles")
# %%
# Obatin important characteristics of the graph
# average node degree
avg_node_degree = np.mean([G.degree(n) for n in G.nodes])
print(f"Average node degree: {avg_node_degree}")
# average clustering coefficient
avg_clustering_coeff = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering_coeff}")
# Average betweenness centrality
avg_betweenness_centrality = np.mean(
    list(nx.betweenness_centrality(G).values()))
print(f"Average betweenness centrality: {avg_betweenness_centrality}")
# Degree assortativity coefficient
assortativity = nx.degree_assortativity_coefficient(G)
print(f"Degree assortativity coefficient: {assortativity}")
# Triadic closure
triadic_closure = nx.transitivity(G)
print(f"Triadic closure: {triadic_closure}")

# Network diameter
# Check for strong connectivity
if nx.is_strongly_connected(G):
    print("Graph is strongly connected.")
    network_diameter = nx.diameter(G)
    print(f"Network diameter: {network_diameter}")
# Check for weak connectivity
elif nx.is_weakly_connected(G):
    print("Graph is weakly connected.")
    network_diameter = nx.diameter(G)
    print(f"Network diameter: {network_diameter}")
    # If neither, then it is not connected
else:
    print("Network diameter cannot be defined (graph is not connected).")

# %%
# Plot the node degree distribution
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure(figsize=(10, 6))
plt.hist(degree_sequence, bins=30, color='blue', alpha=0.7)
plt.title("Node Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# %% Plot the centrality distribution
centrality = nx.betweenness_centrality(G)
plt.figure(figsize=(10, 6))
plt.hist(list(centrality.values()), bins=30, color='blue', alpha=0.7)
plt.title("Betweenness Centrality Distribution")
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
# %%
# show node degrees by color in plot (red = highest node degree, blue = lowest node degree)
plt.figure(figsize=(10, 10))
plt.scatter(
    [data['X'] for _, data in G.nodes(data=True)],
    [data['Y'] for _, data in G.nodes(data=True)],
    c=[G.degree(n) for n in G.nodes()],
    cmap='coolwarm',
    s=50,  # Size of the points
    edgecolor='k',  # Edge color of the points
    alpha=0.7  # Transparency
)
plt.colorbar(label='Node Degree')
plt.title("Node Degree Visualization")
plt.show()


# %%
