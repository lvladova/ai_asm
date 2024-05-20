import streamlit as st
import networkx as nx
import random
import heapq
import graphviz


# Create a synthetic graph
def create_graph(num_nodes):
    """
    Creates a synthetic graph with the specified number of nodes.
    """
    G = nx.DiGraph()
    nodes = [f"Location_{i}" for i in range(num_nodes)]
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            G.add_edge(nodes[i], nodes[j], distance=random.randint(1, 10),
                       traffic=random.randint(1, 5), urgency=random.randint(1, 5))
            G.add_edge(nodes[j], nodes[i], distance=random.randint(1, 10),
                       traffic=random.randint(1, 5), urgency=random.randint(1, 5))
    return G


# Heuristic function for A* algorithm for distance
def heuristic_distance(node, goal, G):
    """
    Heuristic function for A* algorithm based on distance between nodes.
    """
    node_idx = int(node.split('_')[-1])
    goal_idx = int(goal.split('_')[-1])
    return abs(node_idx - goal_idx)


# Traffic heuristic function
def traffic_heuristic(node, goal, G):
    """
    Heuristic function for A* algorithm based on traffic between nodes.
    """
    try:
        shortest_path_length = nx.dijkstra_path_length(G, node, goal, weight='traffic')
    except nx.NetworkXNoPath:
        shortest_path_length = float('inf')
    return shortest_path_length


# Delivery urgency heuristic function
def urgency_heuristic(node, goal, G):
    """
    Heuristic function for A* algorithm based on delivery urgency between nodes.
    """
    try:
        shortest_path_length = nx.dijkstra_path_length(G, node, goal, weight='urgency')
    except nx.NetworkXNoPath:
        shortest_path_length = float('inf')
    return shortest_path_length


# Combined heuristic function
def combined_heuristic(node, goal, G):
    """
    Heuristic function for A* algorithm based on a combination of distance, traffic, and urgency between nodes.
    """
    try:
        shortest_path_length = nx.dijkstra_path_length(G, node, goal, weight=lambda u, v, d: d['distance'] + d['traffic'] + d['urgency'])
    except nx.NetworkXNoPath:
        shortest_path_length = float('inf')
    return shortest_path_length


# A* search algorithm
def a_star_algorithm(G, start, goal, heuristic):
    """
    A* search algorithm for finding the shortest path between two nodes in a graph.
    """
    open_set = []
    heapq.heappush(open_set, (0, start, [start], 0))
    closed_set = set()

    while open_set:
        _, current, path, dist = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == goal:
            return path, dist

        closed_set.add(current)

        for neighbor, attributes in G[current].items():
            if neighbor in closed_set:
                continue
            cost = dist + attributes['distance'] + heuristic(current, neighbor, G)
            heapq.heappush(open_set, (cost, neighbor, path + [neighbor], dist + attributes['distance']))

    return [], 0


def nx_to_dot(G):
    """
    Converts a NetworkX graph to a Graphviz dot object for visualization.
    """
    dot = graphviz.Digraph()
    for node in G.nodes():
        dot.node(node, node)
    for edge in G.edges():
        node1, node2 = edge
        dot.edge(node1, node2,
                 label=f"{G[node1][node2]['distance']} | Traffic: {G[node1][node2]['traffic']} | Urgency: {G[node1][node2]['urgency']}")
    return dot


def calculate_average_traffic_urgency(G, path):
    """
    Calculates the average traffic and urgency along a given path in a graph.
    """
    total_traffic = 0
    total_urgency = 0
    for i in range(len(path) - 1):
        total_traffic += G[path[i]][path[i + 1]]['traffic']
        total_urgency += G[path[i]][path[i + 1]]['urgency']
    average_traffic = total_traffic / (len(path) - 1) if path else 0
    average_urgency = total_urgency / (len(path) - 1) if path else 0
    return average_traffic, average_urgency


def main():
    """
    Main function for running the A* pathfinding visualization.
    """
    graph = create_graph(num_nodes=10)  # This should be part of an initialization that checks session state.
    if 'graph' not in st.session_state:
        st.session_state['graph'] = graph
    else:
        graph = st.session_state['graph']

    if 'dot' not in st.session_state:
        st.session_state['dot'] = nx_to_dot(st.session_state['graph'])    

    st.title("A* Pathfinding Visualization")
    # Display the graph
    st.graphviz_chart(st.session_state['dot'].source)

    # User input for start and goal nodes
    start_node = st.selectbox("Select start location:", list(st.session_state['graph'].nodes()))
    goal_node = st.selectbox("Select goal location:", list(st.session_state['graph'].nodes()))

    # User input for heuristic selection
    heuristic_function = st.selectbox("Select heuristic function:", ['Distance', 'Traffic', 'Urgency', 'Combined'])
    heuristic = {
        'Distance': heuristic_distance,
        'Traffic': traffic_heuristic,
        'Urgency': urgency_heuristic,
        'Combined': combined_heuristic
    }[heuristic_function]

    # Button to find path using A*
    if st.button("Find Path"):
        path, total_distance = a_star_algorithm(st.session_state['graph'], start_node, goal_node, heuristic)
        average_traffic, average_urgency = calculate_average_traffic_urgency(st.session_state['graph'], path)
        st.write(f"Path found: {' -> '.join(path)}")
        st.write(f"Total distance covered: {total_distance}")
        st.write(f"Average traffic along the path: {average_traffic}")
        st.write(f"Average urgency along the path: {average_urgency}")

        # Update the graph to highlight the path
        dot = nx_to_dot(st.session_state['graph'])  # Re-create to reset highlighting
        for i in range(len(path) - 1):
            dot.edge(path[i], path[i + 1], color='red', penwidth='2.0')
        st.graphviz_chart(dot.source)  # Update the graph visualization


if __name__ == "__main__":
    main()
