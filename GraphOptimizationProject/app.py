from flask import Flask, render_template, request, jsonify
import heapq
import time
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyBboxPatch

app = Flask(__name__)

# Ensure static folder exists
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Sample graph data structure
GRAPH = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'D': 5, 'E': 10},
    'C': {'A': 2, 'D': 8, 'F': 10},
    'D': {'B': 5, 'C': 8, 'E': 2, 'F': 6},
    'E': {'B': 10, 'D': 2, 'G': 3},
    'F': {'C': 10, 'D': 6, 'G': 1},
    'G': {'E': 3, 'F': 1}
}

# Heuristic values for A* (straight-line distance to G)
HEURISTIC = {
    'A': 9,
    'B': 7,
    'C': 9,
    'D': 6,
    'E': 3,
    'F': 1,
    'G': 0
}


class AlgorithmRunner:
    """Class to run pathfinding algorithms and track execution time"""
    
    @staticmethod
    def dijkstra(graph, start, end):
        """
        Dijkstra's Algorithm for shortest path
        """
        start_time = time.time()
        
        # Initialize distances and visited set
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Early termination if we reached the end node
            if current_node == end:
                break
            
            # Check all neighbors
            if current_node in graph:
                for neighbor, weight in graph[current_node].items():
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        exec_time = time.time() - start_time
        
        return {
            'path': path if distances[end] != float('inf') else [],
            'cost': distances[end],
            'time': exec_time
        }
    
    @staticmethod
    def a_star(graph, start, end, heuristic):
        """
        A* Algorithm using heuristic values
        """
        start_time = time.time()
        
        # Initialize
        g_score = {node: float('inf') for node in graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph}
        f_score[start] = heuristic[start]
        previous = {node: None for node in graph}
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            
            if current_node in closed_set:
                continue
            
            if current_node == end:
                # Reconstruct path
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
                
                exec_time = time.time() - start_time
                return {
                    'path': path,
                    'cost': g_score[end],
                    'time': exec_time
                }
            
            closed_set.add(current_node)
            
            # Check neighbors
            if current_node in graph:
                for neighbor, weight in graph[current_node].items():
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g_score[current_node] + weight
                    
                    if tentative_g < g_score[neighbor]:
                        previous[neighbor] = current_node
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        exec_time = time.time() - start_time
        return {
            'path': [],
            'cost': float('inf'),
            'time': exec_time
        }
    
    @staticmethod
    def hybrid_algorithm(graph, start, end, heuristic):
        """
        Hybrid Algorithm combining A* and Dijkstra
        Uses A* initially, and when heuristic becomes unreliable (large value),
        switches to Dijkstra for guaranteed optimal path
        """
        start_time = time.time()
        
        # Initialize
        g_score = {node: float('inf') for node in graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph}
        h_start = heuristic.get(start, 0)
        f_score[start] = h_start
        previous = {node: None for node in graph}
        open_set = [(f_score[start], start)]
        closed_set = set()
        nodes_explored = 0
        heuristic_threshold = max(heuristic.values()) * 0.5
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current_node in closed_set:
                continue
            
            if current_node == end:
                # Reconstruct path
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
                
                exec_time = time.time() - start_time
                return {
                    'path': path,
                    'cost': g_score[end],
                    'time': exec_time,
                    'nodes_explored': nodes_explored
                }
            
            closed_set.add(current_node)
            current_h = heuristic.get(current_node, 0)
            
            # Check neighbors
            if current_node in graph:
                for neighbor, weight in graph[current_node].items():
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g_score[current_node] + weight
                    
                    if tentative_g < g_score[neighbor]:
                        previous[neighbor] = current_node
                        g_score[neighbor] = tentative_g
                        
                        # Hybrid logic: use heuristic if reliable, otherwise use g_score
                        neighbor_h = heuristic.get(neighbor, 0)
                        if neighbor_h <= heuristic_threshold:
                            f_score[neighbor] = g_score[neighbor] + neighbor_h
                        else:
                            f_score[neighbor] = g_score[neighbor]  # Dijkstra mode
                        
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        exec_time = time.time() - start_time
        return {
            'path': [],
            'cost': float('inf'),
            'time': exec_time,
            'nodes_explored': nodes_explored
        }


def visualize_graph(start, end, path):
    """
    Visualize the graph and highlight the shortest path
    """
    G = nx.Graph()
    
    # Add edges with weights
    for node, neighbors in GRAPH.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Draw all nodes
    node_colors = []
    for node in G.nodes():
        if node == start:
            node_colors.append('#90EE90')  # Light green for start
        elif node == end:
            node_colors.append('#FFB6C6')  # Light red for end
        elif node in path:
            node_colors.append('#87CEEB')  # Sky blue for path
        else:
            node_colors.append('#D3D3D3')  # Light gray for others
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.9)
    
    # Draw edges
    edge_colors = []
    edge_widths = []
    for edge in G.edges():
        if edge[0] in path and edge[1] in path:
            idx1 = path.index(edge[0])
            idx2 = path.index(edge[1])
            if abs(idx1 - idx2) == 1:
                edge_colors.append('#FF6B6B')  # Red for path edges
                edge_widths.append(3)
            else:
                edge_colors.append('#CCCCCC')
                edge_widths.append(1)
        else:
            edge_colors.append('#CCCCCC')
            edge_widths.append(1)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title('Graph Optimization - Shortest Path Visualization', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    graph_path = os.path.join('static', 'graph.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()


@app.route('/')
def index():
    """Render the main page"""
    nodes = list(GRAPH.keys())
    return render_template('index.html', nodes=nodes)


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    API endpoint to run all three algorithms and return results
    """
    data = request.json
    start = data.get('start')
    end = data.get('end')
    
    # Input validation
    if not start or not end:
        return jsonify({'error': 'Start and end nodes are required'}), 400
    
    if start not in GRAPH or end not in GRAPH:
        return jsonify({'error': 'Invalid node selected'}), 400
    
    if start == end:
        return jsonify({'error': 'Start and end nodes must be different'}), 400
    
    # Run all three algorithms
    runner = AlgorithmRunner()
    
    dijkstra_result = runner.dijkstra(GRAPH, start, end)
    a_star_result = runner.a_star(GRAPH, start, end, HEURISTIC)
    hybrid_result = runner.hybrid_algorithm(GRAPH, start, end, HEURISTIC)
    
    # Visualize the path (use Dijkstra's result as it's guaranteed optimal)
    visualize_graph(start, end, dijkstra_result['path'])
    
    # Prepare comparison data
    results = {
        'dijkstra': {
            'name': "Dijkstra's Algorithm",
            'path': ' → '.join(dijkstra_result['path']) if dijkstra_result['path'] else 'No path found',
            'cost': dijkstra_result['cost'] if dijkstra_result['cost'] != float('inf') else 'N/A',
            'time': f"{dijkstra_result['time']*1000:.4f} ms"
        },
        'a_star': {
            'name': 'A* Algorithm',
            'path': ' → '.join(a_star_result['path']) if a_star_result['path'] else 'No path found',
            'cost': a_star_result['cost'] if a_star_result['cost'] != float('inf') else 'N/A',
            'time': f"{a_star_result['time']*1000:.4f} ms"
        },
        'hybrid': {
            'name': 'Hybrid Algorithm (A* + Dijkstra)',
            'path': ' → '.join(hybrid_result['path']) if hybrid_result['path'] else 'No path found',
            'cost': hybrid_result['cost'] if hybrid_result['cost'] != float('inf') else 'N/A',
            'time': f"{hybrid_result['time']*1000:.4f} ms"
        }
    }
    
    return jsonify({
        'success': True,
        'results': results,
        'graph_image': '/static/graph.png'
    })


@app.route('/api/graph-data', methods=['GET'])
def get_graph_data():
    """Return the graph structure for display"""
    return jsonify({
        'nodes': list(GRAPH.keys()),
        'edges': [
            {'source': node, 'target': neighbor, 'weight': weight}
            for node, neighbors in GRAPH.items()
            for neighbor, weight in neighbors.items()
        ]
    })


if __name__ == '__main__':
    app.run(debug=True)
