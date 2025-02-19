#%%
import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as plt
import numpy as np

# Read the data
river_data = gpd.read_file("C:\\Users\\moham\\OneDrive\\Desktop\\PhD\\Courses\\UWI\\CIE500_MM\\week5\\rivernetwork\\networkoregon.shp")
river_data.plot()

river_data_exploded = river_data.explode(
    ignore_index=True, index_parts=False)[["geometry"]]
river_data_exploded.plot()
river_data_exploded["length"] = river_data_exploded.geometry.length

G = momepy.gdf_to_nx(
    river_data_exploded,
    approach="primal",
    multigraph=False,
    directed=False,
    length="length"
    )

G.remove_edges_from(nx.selfloop_edges(G))
pos = {node: node for node in list(G.nodes())}

fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx(
    G,
    pos=pos,
    width=2,
    with_labels=False,
    node_color="lightblue",
    edge_color="gray",
    node_size=1,
)
ax.axis("off")  # remove the frame of the generated figure
plt.show()

n = len(G.nodes())
m = len(G.edges())
k = 2*m / n
clust = nx.average_clustering(G)
assort_co = nx.degree_assortativity_coefficient(G)
print( n, m, k, clust, assort_co, sep='\n')
# %%
