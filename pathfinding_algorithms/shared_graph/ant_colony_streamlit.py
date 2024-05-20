import streamlit as st
from ant_colony import ant_colony_optimization
from graph_a_start import nx_to_dot
from ga import nx_to_dot as nx_to_dot_ga
from graph_a_start import create_graph


def main():
    num_nodes = 10
    # Create a graph G here
    G = create_graph(num_nodes)
    nodes = list(G.nodes())

    # Create a sidebar for user input
    st.sidebar.title('Ant Colony Optimization Parameters')
    initial_pheromone = st.sidebar.slider('Initial Pheromone', 0.1, 1.0, 0.1)
    alpha = st.sidebar.slider('Alpha', 1, 10, 1)
    beta = st.sidebar.slider('Beta', 1, 10, 1)
    decay_rate = st.sidebar.slider('Decay Rate', 0.1, 1.0, 0.1)
    generations = st.sidebar.slider('Generations', 100, 1000, 100)
    heuristic = st.sidebar.selectbox('Heuristic', ['distance', 'traffic', 'urgency', 'combined'])
    start_node = st.sidebar.selectbox('Start Node', nodes)
    end_node = st.sidebar.selectbox('End Node', nodes)

    st.title("Ant Colony Optimization Pathfinding Visualization")

    # Create a button to start the optimization
    if st.sidebar.button('Start Optimization'):
        dot = nx_to_dot(G)
        st.graphviz_chart(dot.source)
        # Run the ant colony optimization
        best_path, result = ant_colony_optimization(G, initial_pheromone, alpha, beta, decay_rate, generations, heuristic, start_node, end_node)

        # Display the result
        st.write('Best Path: ', best_path)
        st.write('Result: ', result)

        # Display the graph
        dot = nx_to_dot_ga(G, best_path)
        st.graphviz_chart(dot.source)


if __name__ == "__main__":
    main()
