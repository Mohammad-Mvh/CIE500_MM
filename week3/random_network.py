# %%
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# In the following script a random and a distance-based graphs are generated.

## First, 50 random nodes in the horizontal space (0,100) and vertical space (0,100) are created.
random.seed(100)
pos = [(random.random() * 100.0, random.random() * 100.0) for _ in range(50)]

## For the first graph, the edge list is defined based on a given probability for each node pair.
edge_list1 = []
for node_pair in combinations(list(range(50)), 2):
    exist_prob = random.random()
    if exist_prob < 0.4:
        edge_list1.append(node_pair)
    else:
        continue
#print(edge_list)

## For the second graph, the edge list is defined based on a given distance between the all node pairs.
edge_list2 = []
for node_pair in combinations(list(range(50)), 2):
    distance = np.sqrt((pos[node_pair[0]][0] - pos[node_pair[1]][0]) ** 2 + (pos[node_pair[0]][1] - pos[node_pair[1]][1]) ** 2)
    if distance < 30:
        edge_list2.append(node_pair)
    else:
        continue


## The graphs are created (random or distance-based) as directed and undirected.

G1 = nx.from_edgelist(edge_list2, create_using=nx.Graph)
G2 = nx.from_edgelist(edge_list2, create_using=nx.DiGraph)

## Self-loops are added to the directed graph randomly
for node in G2.nodes:
    self_loop_prob = random.random()
    if self_loop_prob < 0.4:
        G2.add_edge(node, node)
    else:
        continue

## Both graphs are plotted.

fig, ax = plt.subplots()
nx.draw_networkx(G1, pos=pos, with_labels=True, ax=ax)
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\Ph.D\\Courses\\AI in UWI\\CIE500_MM\\week3\\Graph.jpg",
    dpi=300,
    bbox_inches="tight",
)


fig, ax = plt.subplots()
nx.draw_networkx(G2, pos=pos, with_labels=True, ax=ax, arrows=True, arrowstyle="<|-")
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\Ph.D\\Courses\\AI in UWI\\CIE500_MM\\week3\\DiGraph.jpg",
    dpi=300,
    bbox_inches="tight",
)

## The directed and undirected graph are compared by checking the adjacency matrices.
A1 = nx.adjacency_matrix(G1).toarray()
A2 = nx.adjacency_matrix(G2).toarray()
D = np.transpose(A2) + A2 - A1 
np.fill_diagonal(D, 0)
# If every element in D is zero, then the directed graph is the same as the undirected graph.
if np.sum(D) == 0:
    print("The second graph is the directed version of first graph.")
else:
    print("The second graph is not the directed version of first graph.")