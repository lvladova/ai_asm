import streamlit as st
import pyvis as pyvis
from pyvis.network import Network
import os
from graph_a_start import create_graph
from ant_colony import ant_colony_optimization
from ga import genetic_algorithm, objective_function_distance, objective_function_traffic, objective_function_urgency, objective_function_combined, nx_to_dot
from graph_a_start import a_star_algorithm, heuristic_distance, traffic_heuristic, urgency_heuristic, combined_heuristic

import streamlit.components.v1 as components

def main():
    # Slider to select the number of nodes
    num_nodes = st.sidebar.slider('Number of Nodes', min_value=2, max_value=100, value=st.session_state.get('num_nodes', 10))

    # Check if the graph already exists and has the correct number of nodes
    if 'graph' in st.session_state and st.session_state['graph'].number_of_nodes() == num_nodes:
        G = st.session_state['graph']
    else:
        # Create the graph with the specified number of nodes
        G = create_graph(num_nodes)
        st.session_state['graph'] = G

    st.session_state['num_nodes'] = num_nodes

    st.title("Path Optimization - Drone Route Planning")

    # Node selection for all algorithms
    start_node = st.sidebar.selectbox("Start Node", list(G.nodes()), key='start_node')
    end_node = st.sidebar.selectbox("End Node", list(G.nodes()), key='end_node')

    # Control panels for each algorithm
    algorithm = st.sidebar.radio("Choose an algorithm", ['All', 'Genetic Algorithm', 'A* Algorithm', 'Ant Colony Optimization'])

    # Initialize paths and colors
    paths = []
    colors = []

    # Genetic Algorithm Parameters
    if algorithm in ['All', 'Genetic Algorithm']:
        popSize = 5
        generations = 10
        ga_objective_function = st.sidebar.selectbox("Objective function for GA", ['Distance', 'Traffic', 'Urgency', 'Combined'])
        if start_node and end_node and start_node in G and end_node in G:
            objective = {
                'Distance': objective_function_distance,
                'Traffic': objective_function_traffic,
                'Urgency': objective_function_urgency,
                'Combined': objective_function_combined
            }[ga_objective_function]
            ga_path, ga_cost = genetic_algorithm(G, popSize, generations, start_node, end_node, objective)
            paths.append(ga_path)
            colors.append('blue')  # GA
            st.write("Best GA route found:", ga_path)
            st.write("GA Cost:", ga_cost)

    # A* Algorithm Parameters
    if algorithm in ['All', 'A* Algorithm']:
        a_star_heuristic_function = st.sidebar.selectbox("Heuristic function for A*", ['Distance', 'Traffic', 'Urgency', 'Combined'])
        heuristic = {
            'Distance': heuristic_distance,
            'Traffic': traffic_heuristic,
            'Urgency': urgency_heuristic,
            'Combined': combined_heuristic
        }[a_star_heuristic_function]
        a_star_path, total_distance = a_star_algorithm(G, start_node, end_node, heuristic)
        print(a_star_path)  # Print the A* path for debugging
        paths.append(a_star_path)
        colors.append('red')  # A*

        # Visualize the A* path alone for debugging
        st.write(f"A* Path found: {' -> '.join(a_star_path)}")
        st.write(f"A* Total distance covered: {total_distance}")

    # Ant Colony Optimization Parameters
    if algorithm in ['All', 'Ant Colony Optimization']:
        initial_pheromone = st.sidebar.slider('Initial Pheromone', 0.1, 1.0, 0.5)
        alpha = st.sidebar.slider('Alpha', 1, 10, 1)
        beta = st.sidebar.slider('Beta', 1, 10, 1)
        decay_rate = st.sidebar.slider('Decay Rate', 0.1, 1.0, 0.5)
        generations_aco = st.sidebar.slider('Generations', 100, 1000, 300)
        heuristic = st.sidebar.selectbox('Heuristic', ['distance', 'traffic', 'urgency', 'combined'])
        aco_path, aco_cost = ant_colony_optimization(G, initial_pheromone, alpha, beta, decay_rate, generations_aco,heuristic, start_node, end_node)
        paths.append(aco_path)
        colors.append('green')  # ACO
        st.write('ACO Best Path:', aco_path)
        st.write('ACO Result:', aco_cost)

    # Visualize all paths
    net = visualize_paths(G, paths, colors)
    colors = ['red', 'blue', 'green']
    graph_path = visualize_paths(G, paths, colors)
    show_graph(graph_path)

    # Display the color legend
    st.markdown("## Color Legend")
    st.markdown("- Genetic Algorithm: blue")
    st.markdown("- A* Algorithm: red")
    st.markdown("- Ant Colony Optimization: green")


def show_graph(graph_path):
    if os.path.exists(graph_path):
        with open(graph_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, width=1200)


def visualize_paths(G, paths, colors):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    
    # Set up the nodes and edges with attributes
    for node in G.nodes():
        net.add_node(node, title=f"Node: {node}")

    # Add edges with tooltips for attributes
    for edge in G.edges(data=True):
        dist = edge[2].get('distance', 'N/A')
        traffic = edge[2].get('traffic', 'N/A')
        urgency = edge[2].get('urgency', 'N/A')
        label = f"Dist: {dist}, Traffic: {traffic}, Urgency: {urgency}"
        color = 'grey'  # Default color
        width = 2  # Default width
        
        # Check if this edge is in any of the paths
        for idx, path in enumerate(paths):
            if (edge[0], edge[1]) in zip(path, path[1:]):
                color = colors[idx % len(colors)]
                width = 5  # Make path edges thicker

        net.add_edge(edge[0], edge[1], title=label, color=color, width=width)

    # Provide options for physics and interaction
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)
    net.show('graph.html')

    return 'graph.html'


if __name__ == "__main__":
    main()

    