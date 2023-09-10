from collections import deque
from utils.graph_visualization import visualize_traversal

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
