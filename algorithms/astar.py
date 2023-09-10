from utils.graph_visualization import visualize_traversal

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