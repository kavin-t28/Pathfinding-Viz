from utils.graph_visualization import visualize_traversal

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
