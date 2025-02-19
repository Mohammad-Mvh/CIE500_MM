#%%
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

## In the following script a random and a distance-based graphs are generated.

# First, 50 random nodes in the horizontal space (0,100) and vertical space (0,100) are created.
random.seed(100)
pos = [(random.random() * 100.0, random.random() * 100.0) for _ in range(50)]

# For the first graph, the edge list is defined based on a given probability for each node pair.
edge_list1 = []
for node_pair in combinations(list(range(50)), 2):
    exist_prob = random.random()
    if exist_prob < 0.4:
        edge_list1.append(node_pair)
    else:
        continue
#print(edge_list)

# For the second graph, the edge list is defined based on a given distance between the all node pairs.
edge_list2 = []
for node_pair in combinations(list(range(50)), 2):
    distance = np.sqrt((pos[node_pair[0]][0] - pos[node_pair[1]][0]) ** 2 + (pos[node_pair[0]][1] - pos[node_pair[1]][1]) ** 2)
    if distance < 30:
        edge_list2.append(node_pair)
    else:
        continue


# The graphs are created (random or distance-based) as directed and undirected.

G1 = nx.from_edgelist(edge_list1, create_using=nx.Graph)
G2 = nx.from_edgelist(edge_list1, create_using=nx.DiGraph)

# Self-loops are added to the directed graph randomly
for node in G2.nodes:
    self_loop_prob = random.random()
    if self_loop_prob < 0.4:
        G2.add_edge(node, node)
    else:
        continue

# Both graphs are plotted.

fig, ax = plt.subplots()
nx.draw_networkx(G1, pos=pos, with_labels=True, ax=ax)
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\PhD\\Courses\\AIinUWI\\CIE500_MM\\NetwokrChar\\Graph.jpg",
    dpi=300,
    bbox_inches="tight",
)


fig, ax = plt.subplots()
nx.draw_networkx(G2, pos=pos, with_labels=True, ax=ax, arrows=True, arrowstyle="<|-")
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\PhD\\Courses\\AIinUWI\\CIE500_MM\\NetwokrChar\\DiGraph.jpg",
    dpi=300,
    bbox_inches="tight",
)

# The directed and undirected graph are compared by checking the adjacency matrices.
A1 = nx.adjacency_matrix(G1).toarray()
A2 = nx.adjacency_matrix(G2).toarray()
D = np.transpose(A2) + A2 - A1 
np.fill_diagonal(D, 0)
# If every element in D is zero, then the directed graph is the same as the undirected graph.
if np.sum(D) == 0:
    print("The second graph is the directed version of first graph.")
else:
    print("The second graph is not the directed version of first graph.")


## Now, the network characteristics of the random network are calculated.

# First, network size:
node_size = G1.number_of_nodes()
print(f"The size of the network is: {node_size}")
edge_size = G1.number_of_edges()
print(f"The size of the network is: {edge_size}")

# Second,  network diameter:
diameter = nx.diameter(G1)
print(f"The diameter of the network is: {diameter}")

# Third, network average shortest path length:
avg_shortest_path = nx.average_shortest_path_length(G1)
print(f"The average shortest path length of the network is: {avg_shortest_path}")

# Fourth, network connectivity:
connectivity = nx.is_connected(G1)
print(f"The network is connected: {connectivity}")

# Fifth, a statistical analysis of nodes' degrees and its distribution:
degree = dict(G1.degree())
degree_values = list(degree.values())
mean_degree = np.mean(degree_values)
std_degree = np.std(degree_values)
print(f"The mean degree of the network is: {mean_degree}")
print(f"The standard deviation of the network is: {std_degree}")

# The degree distribution is plotted.
fig, ax = plt.subplots()
plt.hist(degree_values, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.tight_layout()
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\PhD\\Courses\\AIinUWI\\CIE500_MM\\NetwokrChar\\DegreeDistribution.jpg",
    dpi=300,
    bbox_inches="tight",
)

# Sixth, betweenness centrality and average betweenness centrality:
betweenness = nx.betweenness_centrality(G1)
max_betweenness = max(betweenness.values())
min_betweenness = min(betweenness.values())
avg_betweenness = np.mean(list(betweenness.values()))
print(f"The maximum betweenness centrality is: {max_betweenness}")
print(f"The minimum betweenness centrality is: {min_betweenness}")
print(f"The average betweenness centrality is: {avg_betweenness}")

# Seventh, closeness centrality and average closeness centrality:
closeness = nx.closeness_centrality(G1)
max_closeness = max(closeness.values())
min_closeness = min(closeness.values())
avg_closeness = np.mean(list(closeness.values()))
print(f"The maximum closeness centrality is: {max_closeness}")
print(f"The minimum closeness centrality is: {min_closeness}")
print(f"The average closeness centrality is: {avg_closeness}")

# Last, the topological order of the directed version network nodes:
# remove self loops:
G2.remove_edges_from(nx.selfloop_edges(G2))
# find the topological order:
topological_order = list(nx.topological_sort(G2))
print(f"The topological order of the network is: {topological_order}")
# %%
