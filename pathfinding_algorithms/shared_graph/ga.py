import pandas as pd
import random
import math
import numpy as np
import streamlit as st
import graphviz
from graph_a_start import create_graph

popSize = 5  # population size
elite_size = 5
generations = 10


def nx_to_dot(G, best_path):
    dot = graphviz.Digraph()
    for node in G.nodes():
        dot.node(node, node)
    for edge in G.edges():
        node1, node2 = edge
        if best_path and (node1, node2) in zip(best_path, best_path[1:]):
            dot.edge(node1, node2, color='red', penwidth='2.0',
                     label=f"{G[node1][node2]['distance']} | Traffic: {G[node1][node2]['traffic']} | Urgency: {G[node1][node2]['urgency']}")
        else:
            dot.edge(node1, node2,
                     label=f"{G[node1][node2]['distance']} | Traffic: {G[node1][node2]['traffic']} | Urgency: {G[node1][node2]['urgency']}")
    return dot


def get_best_distance(start_indx, end_indx):
    #this function gets the best distance from the genetic algorithm
    result = genetic_algorithm(start_indx, end_indx)
    best_individual = result['best_individual']
    shortest_distance = best_individual[0]

    return shortest_distance


def genetic_algorithm(G, popSize, generations, start_node, end_node, objective_function):
    population = generate_population(G, popSize, start_node, end_node)

    scores = calculate_fitness(G, population, objective_function)

    for _ in range(generations):
        parents = make_selection(population, scores)
        offspring = breed_population(parents, elite_size)
        population = mutate_population(offspring, 0.5, elite_size)
        scores = calculate_fitness(G, population, objective_function)

    best_individual = get_best_individual(scores, population)
    return best_individual[0], best_individual[1]


def calculate_fitness(G, population, objective_function):
    fitness = []
    for path in population:
        cost = objective_function(G, path)
        if cost == 0:
            fitness.append((path, float('inf')))  # Avoid division by zero
        else:
            fitness.append((path, 1 / cost))
    return fitness


def get_best_individual(scores):
    best_index = 0
    highest_score = scores[0][1] if scores[0][1] != float('inf') else float(1e9)
    for i in range(1, len(scores)):
        score = scores[i][1] if scores[i][1] != float('inf') else float(1e9)
        if score > highest_score:
            highest_score = score
            best_index = i
    best_route = scores[best_index][0]
    return best_route


def generate_random_path(G, start_node, end_node):
    path = list(G.nodes())
    if start_node in path:
        path.remove(start_node)
    else:
        print(f"Start node {start_node} not in path")
    if end_node in path:
        path.remove(end_node)
    else:
        print(f"End node {end_node} not in path")

    subset_size = random.randint(2, len(path))  # Adjust range as needed
    path = random.sample(path, subset_size)

    path.insert(0, start_node)
    path.append(end_node)
    return path


def get_best_individual(scores, population):
    best_index = 0
    lowest_score = scores[0][1] if scores[0][1] != float('inf') else float(1e9)

    for i in range(1, len(scores)):
        score = scores[i][1] if scores[i][1] != float('inf') else float(1e9)
        if score < lowest_score:
            lowest_score = score
            best_index = i

    return population[best_index], 1 / lowest_score


def next_generation(currentGen, eliteSize, mutationRate):
    #this function creates the next generation
    popRanked = calculate_fitness(currentGen)
    selectionResults = make_selection(currentGen, popRanked)
    matingpool = mating_pool(currentGen, selectionResults)
    children = breed_population(matingpool, eliteSize)
    nextGeneration = mutate_population(children, mutationRate, eliteSize)

    return nextGeneration


def mutate_population(population, mutationRate, eliteSize):
    #this function mutates the population
    mutatedPop = []

    for ind in range(0, len(population)):
        # create mutation just if it is not the elite
        if ind >= eliteSize:
            mutatedInd = mutation(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        else:
            mutatedPop.append(population[ind])

    return mutatedPop


def mutation(order, mutation_rate):
    #this function mutates the order of the cities with a mutation rate, returns the new order

    if random.random() < mutation_rate:
        order_list = list(order)
        idx1 = random.randint(0, len(order)-1)
        idx2 = random.randint(0, len(order)-1)

        city1 = order_list[idx1]
        city2 = order_list[idx2]

        order_list[idx1] = city2
        order_list[idx2] = city1

        order = tuple(order_list)

    return order


def breed_population(mating_pool, elite_size):
    """
        this function breeds the population with the mating pool and the elite size and returns the children
        usng the davis crossover function for the crossover of the parents
    """
    children = []

    elite_size = min(elite_size, len(mating_pool))

    for i in range(elite_size):
        children.append(mating_pool[i])

    for i in range(elite_size, len(mating_pool), 2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[len(mating_pool)-i-1]

        child1, child2 = davis_crossover(parent1, parent2)
        children.append(child1)
        children.append(child2)

    return children


def mating_pool(population, selection_results):
    """
    create a mating pool based on the selection results
    inpiration: https://www.youtube.com/watch?v=ETphJASzYes

    """
    matin_gpool = []
    for result in selection_results:
        index = result[0]

        if isinstance(index, np.ndarray):
            index = index[0]
        if index < len(population):
            index = index.astype(int)
            matin_gpool.append(population[index])
    random.shuffle(matin_gpool)

    return matin_gpool


def make_selection(population, fitness):
    """
        this function uses elitism and roulette wheel selection to select the individuals for the next generation
        inspiration from:
        https://www.youtube.com/@TheCodingTrain
        https://www.youtube.com/watch?v=M3KTWnTrU_c
        https://www.youtube.com/watch?v=0z82YOXlIiE&list=PLPbG9ouofIFfhb47dPnRRoWdDYtjVM0vw&index=15
        https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/
    """
    elite_size = 5
    selection_results = []

    df_population = pd.DataFrame(fitness, columns=["Individual", "Fitness"])
    #using pandas dataframe to sort the population by fitness
    df_population.sort_values(by="Fitness", ascending=False, inplace=True)
    df_population.reset_index(drop=True, inplace=True)

    df_population['cum_sum'] = df_population['Fitness'].cumsum()
    df_population['cum_perc'] = 100 * \
        df_population['cum_sum'] / df_population['Fitness'].sum()

    elite_individuals = df_population.loc[:elite_size-1, 'Individual'].tolist()
    #selecting elite individuals
    selection_results.extend(elite_individuals)

    for _ in range(len(population) - elite_size):
    #selecting remaining individuals using roulette wheel selection
        pick = 100 * random.random()
        for i in range(len(population)):
            if pick <= df_population.loc[i, 'cum_perc']:
                selection_results.append(df_population.loc[i, 'Individual'])
                break

    return selection_results


def davis_crossover(parent_a, parent_b):
    """
        davies order crossover is a type of crossover that is used to create offsprings from two parents
        inspiration from:
        https://github.com/marcosdg/davis-order-crossover/blob/master/davis_crossover.py
        https://www.redalyc.org/pdf/2652/265219618002.pdf
        https://www.researchgate.net/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators
    """
    length = len(parent_a)
    start_idx = random.randint(0, length-1)
    end_idx = random.randint(start_idx, length-1)

    pool = np.concatenate((parent_a, parent_b))
    offspring1 = np.zeros_like(parent_a)
    offspring2 = np.zeros_like(parent_a)

    crossover_section = parent_b[start_idx:end_idx+1]
    offspring1[start_idx:end_idx+1] = crossover_section

    random_choice_offspring(pool, offspring1, start_idx, end_idx)

    # repeat to generate second offspring
    crossover_section = parent_a[start_idx:end_idx+1]
    offspring2[start_idx:end_idx+1] = crossover_section

    avail = ~np.isin(pool, np.concatenate((offspring1, offspring2)))
    for i in range(len(offspring2)):
        if i < start_idx or i > end_idx:
            # update avail inside the loop
            avail = ~np.isin(pool, np.concatenate((offspring1, offspring2)))
            if np.any(avail):  # check if pool[avail] is not empty
                chosen = np.random.choice(pool[avail])
                offspring2[i] = chosen
                pool = pool[pool != chosen]  # remove chosen element from pool
            else:
                break

    avail = ~np.isin(pool, np.concatenate(
        (offspring1, offspring2)))  # update avail

    return offspring1, offspring2


def random_choice_offspring(pool, offspring1, start_idx, end_idx):
    # https://numpy.org/doc/stable/reference/generated/numpy.isin.html
    avail = ~np.isin(pool, offspring1)
    for i in range(len(offspring1)):
        if i < start_idx or i > end_idx:
            if np.any(avail):
                offspring1[i] = np.random.choice(pool[avail])
            else:
                break


def calculate_fitness(G, population, objective_function):
    fitness = []
    for path in population:
        cost = objective_function(G, path)
        if cost == 0:
            fitness.append((path, float('inf')))  # Avoid division by zero
        else:
            fitness.append((path, 1 / cost))
    return fitness


def generate_population(G, popSize, start_node, end_node):
    return [generate_random_path(G, start_node, end_node) for _ in range(popSize)]


def generate_random_path(G, start_node, end_node):
    # Generates a random path using graph nodes
    path = list(G.nodes())
    if start_node in path:
        path.remove(start_node)
    else:
        print(f"Start node {start_node} not in path")
    if end_node in path:
        path.remove(end_node)
    else:
        print(f"End node {end_node} not in path")

    # Randomly select a subset of nodes
    subset_size = random.randint(2, len(path))  # Adjust range as needed
    path = random.sample(path, subset_size)

    path.insert(0, start_node)
    path.append(end_node)
    return path


def objective_function_distance(G, path):
    # Calculates the total cost of a path based on graph attributes
    total_cost = 0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_cost += edge_data['distance']
    return total_cost


def objective_function_traffic(G, path):
    # Calculates the total traffic of a path based on graph attributes
    total_traffic = 0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_traffic += edge_data['traffic']
    return total_traffic


def objective_function_urgency(G, path):
    # Calculates the total urgency of a path based on graph attributes
    total_urgency = 0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_urgency += edge_data['urgency']
    return total_urgency


def objective_function_combined(G, path):
    # Calculates the total cost of a path based on all graph attributes
    total_cost = 0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_cost += edge_data['distance'] + edge_data['traffic'] + edge_data['urgency']
    return total_cost


def euclidean_distance(pointa, pointb):
    return math.sqrt(
        math.pow(pointa[0] - pointb[0], 2) +
        math.pow(pointa[1] - pointb[1], 2)
    )


def main():
    G = create_graph()  # This should be part of an initialization that checks session state.
    if 'graph' not in st.session_state:
        st.session_state['graph'] = G
    else:
        G = st.session_state['graph']

    city_locations = list(G.nodes(data='pos'))

    st.title("Drone Route Optimization with Genetic Algorithm")

    # User input for start and goal nodes
    start_node = st.selectbox("Select start location:", list(G.nodes()))
    end_node = st.selectbox("Select goal location:", list(G.nodes()))

    # User input for objective function selection
    objective_function_name = st.selectbox("Select objective function:", ['Distance', 'Traffic', 'Urgency', 'Combined'])
    objective_function = {
        'Distance': objective_function_distance,
        'Traffic': objective_function_traffic,
        'Urgency': objective_function_urgency,
        'Combined': objective_function_combined
    }[objective_function_name]

    # Display the initial graph
    dot = nx_to_dot(G, city_locations)
    st.graphviz_chart(dot.source)

    if st.button("Run Genetic Algorithm"):
        best_route, cost = genetic_algorithm(G, popSize, generations, start_node, end_node, objective_function)
        st.write("Best route found:", best_route)
        st.write("Cost:", cost)
        dot = nx_to_dot(G, best_route)
        st.graphviz_chart(dot.source)


if __name__ == "__main__":
    main()
