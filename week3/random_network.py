# %%
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# In the following script we are going to generate a random graph

## First, we generate 8 random nodes in the horizontal space (0,2) and vertical space (0,4)
random.seed(66)
pos = [(random.random() * 10.0, random.random() * 10.0) for _ in range(10)]

## Second, we create an edge list by a given probability.
edge_list = []
for node_pair in combinations(list(range(10)), 2):
    exist_prob = random.random()
    if exist_prob < 0.4:
        edge_list.append(node_pair)
    else:
        continue
print(edge_list)
## Now, we can create the graph based on the positions and edge list.

G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

# Now let's add a self-loop to the network
G.add_edge(4, 4)
G.add_edge(2, 2)

G.add_edge(1,9)
G.add_edge(9,1)

fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax)
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\Ph.D\\Courses\\AI in UWI\\CIE500_MM\\week3\\examplegraph.jpg",
    dpi=300,
    bbox_inches="tight",
)




A = nx.adjacency_matrix(G).toarray()
print(A)



fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax, arrows=True, arrowstyle="<|-", style="dashed")
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "C:\\Users\\moham\\OneDrive\\Desktop\\Ph.D\\Courses\\AI in UWI\\CIE500_MM\\week3\\examplegraph2.jpg",
    dpi=300,
    bbox_inches="tight",
)

# Finally, we can get the edgelist and adjancency matrix from Graph directly.

#print(f"The adjancency matrix of G is \n {nx.adjacency_matrix(G).toarray()}")

#print(f"The edge list of G is \n {nx.to_edgelist(G)}")
# %%
