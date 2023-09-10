import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

def visualize_traversal(graph, traversal_path, title, shade_edge=None, shade_node=None):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    
    plt.ion()  # Turn on interactive mode
    
    for i, node in enumerate(traversal_path):
        nx.draw(graph, pos, node_color='lightblue', with_labels=True)

        if shade_edge and i < len(traversal_path) - 1:
            edge = (node, traversal_path[i + 1])
            nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color='r', width=2)
        
        if shade_node and node == shade_node:
            nx.draw(graph, pos, nodelist=[node], node_color='g', with_labels=True)

        plt.title(f"{title} - Step {i+1}")
        plt.pause(0.5)  # Short pause between frames
        plt.clf()

    plt.ioff()  # Turn off interactive mode
    plt.show()

