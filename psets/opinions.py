import numpy as np
import networkx as nx

G = nx.karate_club_graph() # Load a graph from networkx
# You can choose your own from the this webpage: https://networkx.org/documentation/stable/auto_examples/graph/index.html

pos = nx.spring_layout(G) # This fixes the position of the edges for drawing consistently

n = len(G) # Compute the number of nodes

opinions = np.random.normal(size=n) # Initialize random starting opinions

nx.draw(G, node_color=opinions, cmap='viridis', pos=pos) # Plot opinions by node

