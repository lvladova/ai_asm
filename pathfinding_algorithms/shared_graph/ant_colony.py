import numpy as np
import random

def initialize_pheromones(G, initial_pheromone):
    """
    Initialize the pheromone levels for all edges in the graph.

    Args:
        G (networkx.Graph): The graph.
        initial_pheromone (float): The initial pheromone level.

    """
    for u, v, d in G.edges(data=True):
        d['pheromone'] = initial_pheromone


def choose_next_node(G, current_node, alpha, beta, heuristic):
    """
    Choose the next node to visit based on the current node, pheromone levels, and heuristic information.

    Args:
        G (networkx.Graph): The graph.
        current_node: The current node.
        alpha (float): The pheromone influence parameter.
        beta (float): The heuristic influence parameter.
        heuristic (str): The type of heuristic to use.

    Returns:
        The next node to visit.

    """
    edges = G.edges(current_node, data=True)
    pheromones = np.array([edge_data['pheromone'] ** alpha for _, _, edge_data in edges])
    if heuristic == 'distance':
        heuristic_values = np.array([1.0 / edge_data['distance'] ** beta for _, _, edge_data in edges])
    elif heuristic == 'traffic':
        heuristic_values = np.array([1.0 / edge_data['traffic'] ** beta for _, _, edge_data in edges])
    elif heuristic == 'urgency':
        heuristic_values = np.array([1.0 / edge_data['urgency'] ** beta for _, _, edge_data in edges])
    elif heuristic == 'combined':
        heuristic_values = np.array([(1.0 / edge_data['distance'] ** beta + 1.0 / edge_data['traffic'] ** beta + 1.0 / edge_data['urgency'] ** beta) / 3 for _, _, edge_data in edges])
    probabilities = pheromones * heuristic_values
    probabilities /= probabilities.sum()
    next_node = random.choices([v for _, v, _ in edges], weights=probabilities)[0]
    return next_node


def update_pheromones(G, ants, decay_rate, additional_pheromone):
    """
    Update the pheromone levels based on the paths taken by the ants.

    Args:
        G (networkx.Graph): The graph.
        ants (list): List of paths taken by the ants.
        decay_rate (float): The rate at which pheromones evaporate.
        additional_pheromone (float): The amount of additional pheromone to deposit.

    """
    # Evaporate pheromones
    for u, v, d in G.edges(data=True):
        d['pheromone'] *= (1 - decay_rate)
    
    # Add new pheromones
    for path, path_cost in ants:
        for i in range(len(path)-1):
            if path_cost > 0:
                G[path[i]][path[i+1]]['pheromone'] += additional_pheromone / path_cost


def ant_colony_optimization(G, initial_pheromone=1.0, alpha=1, beta=1, decay_rate=0.1, generations=100, heuristic='distance', start_node=None, end_node=None):
    """
    Perform ant colony optimization to find the best path in the graph.

    Args:
        G (networkx.Graph): The graph.
        initial_pheromone (float): The initial pheromone level.
        alpha (float): The pheromone influence parameter.
        beta (float): The heuristic influence parameter.
        decay_rate (float): The rate at which pheromones evaporate.
        generations (int): The number of generations to run the algorithm.
        heuristic (str): The type of heuristic to use.
        start_node: The starting node. If not specified, a random node will be chosen.
        end_node: The ending node.

    Returns:
        The best path and its cost.

    """
    # Initialize pheromones
    initialize_pheromones(G, initial_pheromone)

    for _ in range(generations):
        ants = []
        for _ in range(G.number_of_nodes()):
            # Start from the start_node if it's specified
            path = [start_node if start_node else random.choice(list(G.nodes()))]
            for _ in range(G.number_of_nodes() - 1):
                new_node = choose_next_node(G, path[-1], alpha, beta, heuristic)
                path.append(new_node)
            path_cost = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
            ants.append((path, path_cost))

        update_pheromones(G, ants, decay_rate, additional_pheromone=100)

    # Find the best path from the pheromone levels
    best_path = None
    best_cost = float('inf')
    for node in G.nodes():
        # Start from the start_node if it's specified
        path = [start_node if start_node else node]
        for _ in range(G.number_of_nodes() - 1):
            new_node = choose_next_node(G, path[-1], alpha, beta, heuristic)
            path.append(new_node)
        path_cost = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
        if path_cost < best_cost:
            best_cost = path_cost
            best_path = path

    return best_path, best_cost
