import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt

# 1. Create a sample Graph using NetworkX
# Let's create a 'Barbell' graph (two cliques connected by a path)
G_nx = nx.barbell_graph(m1=10, m2=4)
print(G_nx)

# 2. Convert NetworkX graph to iGraph (required by leidenalg)
# leidenalg works natively with igraph for speed
G_ig = ig.Graph.from_networkx(G_nx)

# 3. Run the Leiden Algorithm
# We optimize for Modularity here
partition = la.find_partition(G_ig, la.ModularityVertexPartition)

# 4. Extract results
# Map community IDs back to the nodes
community_map = {}
for i, community in enumerate(partition):
    for node_index in community:
        community_map[node_index] = i

# 5. Visualization
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_nx)

# Draw nodes colored by their community
colors = [community_map[node] for node in G_nx.nodes()]
nx.draw(G_nx, pos, node_color=colors, with_labels=True, 
        cmap=plt.cm.rainbow, node_size=500, edge_color='gray')

plt.title("Leiden Community Detection")
plt.show()

print(f"Number of communities found: {len(partition)}")
