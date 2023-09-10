import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from algorithms import bfs ,dfs, astar

# Function to visualize the graph
def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

# Function to visualize the graph traversal
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





if __name__ == "__main__":
    G = nx.Graph()
    
    # User input for graph edges and edge weights
    num_edges = int(input("Enter the number of edges: "))
    for _ in range(num_edges):
        edge = tuple(map(int, input("Enter an edge (node1 node2 weight): ").split()))
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    visualize_graph(G)
    
    # User input for algorithms
    algorithms = ['BFS', 'DFS', 'A*']
    chosen_algorithm = input(f"Choose an algorithm ({', '.join(algorithms)}): ").upper()
    
    start_node = int(input("Enter the starting node: "))
    dest_node = int(input("Enter the destination node: "))
    
    # User input for heuristic values (for A* algorithm)
    heuristic = {}
    for node in G.nodes():
        heuristic[node] = float(input(f"Enter the heuristic value for node {node}: "))
    
    if chosen_algorithm == 'BFS':
        path = bfs(G, start_node, dest_node)
    elif chosen_algorithm == 'DFS':
        path = dfs(G, start_node, dest_node)
    elif chosen_algorithm == 'A*':
        path = astar(G, start_node, dest_node, heuristic)

    else:
        print("Invalid algorithm choice.")
        exit()
    
    if path:
        print(f"Path from {start_node} to {dest_node}:", path)
    else:
        print(f"No path found from {start_node} to {dest_node}")
