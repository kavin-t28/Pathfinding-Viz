import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Function to visualize the graph
def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

# Function to visualize a path
def visualize_path(graph, path):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    
    nx.draw(graph, pos, node_color='lightblue', with_labels=True)
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color='r', width=2)
    
    plt.title("British Museum Search - Path")
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
        # plt.clf()

      # Turn off interactive mode
    plt.show()

# British Museum Search (Exhaustive Search)
def british_museum_search(graph, start_node, dest_node):
    def explore_path(node, path):
        if node == dest_node:
            all_paths.append(path.copy())
            return
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_path = path + [neighbor]
                explore_path(neighbor, new_path)

    all_paths = []
    explore_path(start_node, [start_node])
    return all_paths


# Breadth-First Search
def bfs(graph, start_node, dest_node):
    queue = deque([(start_node, [start_node])])

    while queue:
        node, path = queue.popleft()

        if node == dest_node:
            visualize_traversal(graph, path, "Breadth-First Search", shade_node=dest_node)
            return path
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
                visualize_traversal(graph, path + [neighbor], "Breadth-First Search", shade_edge=(node, neighbor))

    return None

# Depth-First Search
def dfs(graph, start_node, dest_node):
    stack = [(start_node, [start_node])]

    while stack:
        node, path = stack.pop()

        if node == dest_node:
            visualize_traversal(graph, path, "Depth-First Search", shade_node=dest_node)
            return path
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
                visualize_traversal(graph, path + [neighbor], "Depth-First Search", shade_edge=(node, neighbor))

    return None

# Beam Width Search
def beam_search(graph, start_node, dest_node, beam_width):
    queue = [(start_node, [start_node])]

    while queue:
        next_queue = []

        for node, path in queue:
            if node == dest_node:
                visualize_traversal(graph, path, "Beam Width Search", shade_node=dest_node)
                return path

            for neighbor in graph.neighbors(node):
                if neighbor not in path:
                    next_queue.append((neighbor, path + [neighbor]))
                    visualize_traversal(graph, path + [neighbor], "Beam Width Search", shade_edge=(node, neighbor))

            next_queue.sort(key=lambda x: len(x[1]))  # Sort by path length
            queue = next_queue[:beam_width]

    return None

# Branch and Bound Search
def branch_and_bound(graph, start_node, dest_node):
    priority_queue = [(0, [start_node])]  # (lower bound, path)
    best_cost = float('inf')
    best_path = None

    while priority_queue:
        lower_bound, path = priority_queue.pop(0)

        if lower_bound > best_cost:
            continue

        node = path[-1]

        if node == dest_node:
            visualize_traversal(graph, path, "Branch and Bound Search", shade_node=dest_node)
            return path

        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                new_path = path + [neighbor]
                lower_bound = lower_bound + edge_weight
                priority_queue.append((lower_bound, new_path))
                priority_queue.sort()  # Sort by lower bound
                visualize_traversal(graph, new_path, "Branch and Bound Search", shade_edge=(node, neighbor))

    return None

# A* Search
def astar(graph, start_node, dest_node, heuristic):
    open_list = [(0 + heuristic[start_node], [start_node])]  # (f-cost, path)
    closed_set = set()

    while open_list:
        open_list.sort(key=lambda x: x[0])  # Sort by f-cost
        f, path = open_list.pop(0)
        node = path[-1]

        if node == dest_node:
            visualize_traversal(graph, path, "A* Search", shade_node=dest_node)
            return path

        if node in closed_set:
            continue
        
        closed_set.add(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in closed_set:
                edge_weight = graph[node][neighbor]['weight']
                new_path = path + [neighbor]
                g_cost = len(new_path) - 1  # Path length is the g-cost
                f_cost = g_cost + heuristic[neighbor]
                open_list.append((f_cost, new_path))
                visualize_traversal(graph, new_path, "A* Search", shade_edge=(node, neighbor))

    return None

# Oracle Search
def oracle_search(graph, start_node, dest_node, oracle_function):
    stack = [(start_node, [start_node])]

    while stack:
        node, path = stack.pop()

        if node == dest_node:
            visualize_traversal(graph, path, "Oracle Search", shade_node=dest_node)
            return path
        
        oracle_result = oracle_function(node)
        
        if not oracle_result:
            continue
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
                visualize_traversal(graph, path + [neighbor], "Oracle Search", shade_edge=(node, neighbor))

    return None

# Best-First Search
def best_first_search(graph, start_node, dest_node, heuristic):
    queue = [(heuristic[start_node], [start_node])]

    while queue:
        h, path = heapq.heappop(queue)
        node = path[-1]

        if node == dest_node:
            visualize_path(graph, path)
            return path

        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                heapq.heappush(queue, (heuristic[neighbor], path + [neighbor]))
    
    return None

# AO* Search
def ao_star(graph, start_node, dest_node, heuristic):
    while True:
        path = astar(graph, start_node, dest_node, heuristic)
        
        if path is None:
            print("No path found.")
            return
        
        # Update heuristic values based on path cost
        for node in graph.nodes():
            if node in path:
                heuristic[node] = len(path) - path.index(node)
        
        # Check if heuristic values have converged (optional)
        # If they have converged, break the loop
        
        # Print the current heuristic values (optional)
        print("Current heuristic values:", heuristic)
        
        # Break the loop (optional)
        break
    
    return path

# Branch and Bound Search
def branch_and_bound(graph, start_node, dest_node):
    priority_queue = [(0, [start_node])]  # (lower bound, path)
    best_cost = float('inf')
    best_path = None

    while priority_queue:
        lower_bound, path = heapq.heappop(priority_queue)

        if lower_bound > best_cost:
            continue

        node = path[-1]

        if node == dest_node:
            best_cost = lower_bound
            best_path = path
            continue

        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                new_path = path + [neighbor]
                new_lower_bound = lower_bound + edge_weight
                heapq.heappush(priority_queue, (new_lower_bound, new_path))
                visualize_traversal(graph, new_path, "Branch and Bound Search", shade_edge=(node, neighbor))

    return best_path


if __name__ == "__main__":
    G = nx.Graph()
    
    # User input for graph edges and edge weights
    num_edges = int(input("Enter the number of edges: "))
    for _ in range(num_edges):
        edge = tuple(map(int, input("Enter an edge (node1 node2 weight): ").split()))
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    visualize_graph(G)
    
    # User input for algorithms
    algorithms = ['BFS', 'DFS', 'A*','BEAM','BRANCH','ORACLE','BMS','BEST','AO*']
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

    elif chosen_algorithm == 'BEAM':
        beam_width = int(input("Enter the beam width: "))
        path = beam_search(G, start_node, dest_node, beam_width)

    elif chosen_algorithm == 'BRANCH':
        path = branch_and_bound(G, start_node, dest_node)

    elif chosen_algorithm == 'ORACLE':
        oracle_result = {}  # Store oracle results for nodes
        oracle_function = lambda node: oracle_result.get(node, False)
        path = oracle_search(G, start_node, dest_node, oracle_function)


    elif chosen_algorithm == 'BMS':
        paths = british_museum_search(G, start_node, dest_node)
        if paths:
            print(f"All possible paths from {start_node} to {dest_node}:")
            for path in paths:
                print(path)
                visualize_path(G, path)
        else:
            print(f"No path found from {start_node} to {dest_node}")

    elif chosen_algorithm == 'BEST':
        path = best_first_search(G, start_node, dest_node, heuristic)

    elif chosen_algorithm == 'AO*':
        path = ao_star(G, start_node, dest_node, heuristic)
    else:
        print("Invalid algorithm choice.")
        exit()
    
    if path:
        print(f"Path from {start_node} to {dest_node}:", path)
    else:
        print(f"No path found from {start_node} to {dest_node}")