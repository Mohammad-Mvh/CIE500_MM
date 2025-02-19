# %%

import networkx as nx
import matplotlib.pyplot as plt
import copy

# Create an empty graph
G = nx.DiGraph()

# Larger connected component (4 nodes)
G.add_weighted_edges_from([(2, 1, 1), (2, 3, 3), (3, 1, 2)])

# Smaller connected component (3 nodes)
G.add_weighted_edges_from([(5, 4, 1), (5, 6, 2), (6, 4, 3)])

# Connect two components
G.add_weighted_edges_from([(3, 6, 1)])

pos = nx.circular_layout(G)

# Draw the graph without weight
fig, ax = plt.subplots(figsize=(5, 8))
nx.draw_networkx(
    G,
    pos=pos,
    width=1,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=1000,
    font_size=16,
)
ax.axis("off")  # remove the frame of the generated figure

plt.savefig(
    "/Users/xudongfan/Documents/Courses/CIE500-UrbanNetworks/slides/week4/Example.jpg",
    dpi=600,
    bbox_inches="tight",
)
plt.show()


# Draw the graph with weight
weights = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}

weights_list = [d["weight"] for u, v, d in G.edges(data=True)]

fig, ax = plt.subplots(figsize=(5, 8))

nx.draw_networkx(
    G,
    pos=pos,
    width=weights_list,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=1000,
    font_size=16,
)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=weights)
ax.axis("off")  # remove the frame of the generated figure

plt.savefig(
    "/Users/xudongfan/Documents/Courses/CIE500-UrbanNetworks/slides/week4/Example_weights.jpg",
    dpi=600,
    bbox_inches="tight",
)
plt.show()


print(
    f"The adjancency matrix of G is \n {nx.adjacency_matrix(G, nodelist=list(range(1,7))).toarray()}"
)

# get the topological sort order of the graph.

sorted_order = list(nx.topological_sort(G))

print(f"the sorted order is {sorted_order}")

print(
    f"the lenght of sorted order is {len(sorted_order)}\n the total number of nodes is {len(G.nodes())}"
)

# the following codes calculate the betwenness centrality of all nodes of graph G.

node_central = nx.betweenness_centrality(G)

fig, ax = plt.subplots(figsize=(5, 8))

nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    node_color=list(node_central.values()),
    edge_color="gray",
    node_size=1000,
    font_size=16,
)
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "/Users/xudongfan/Documents/Courses/CIE500-UrbanNetworks/slides/week4/node_betweenness.jpg",
    dpi=600,
    bbox_inches="tight",
)
plt.show()


node_clossness = nx.closeness_centrality(G)

fig, ax = plt.subplots(figsize=(5, 8))

nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    node_color=list(node_clossness.values()),
    edge_color="gray",
    node_size=1000,
    font_size=16,
)
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "/Users/xudongfan/Documents/Courses/CIE500-UrbanNetworks/slides/week4/node_closeness.jpg",
    dpi=600,
    bbox_inches="tight",
)
plt.show()
