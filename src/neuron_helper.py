import numpy as np

from plot import plot_route_m

def select_closest_neuron(candidates, origin, network_inhibit):
    min = 100000
    id = -1
    for idx, neuron in enumerate(candidates):
        if network_inhibit[idx]:
            continue
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            id = idx
    return id

def select_closest_in_cluster(cluster, origin, network):
    min = 100000
    closest_neuron = None
    id = -1
    for idx, neuron in enumerate(cluster):
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            closest_neuron = neuron
    for idx, network_neuron in enumerate(network):
        if np.array_equal(network_neuron, closest_neuron):
            id = idx
            break
    return id

def select_closest_for_c(network, c):
    min = 100000
    winner_idx = -1
    for idx, neuron in enumerate(network):
        if np.linalg.norm(neuron-c) < min:
            min = np.linalg.norm(neuron-c)
            winner_idx = idx

    return winner_idx

def select_closest_neuron_for_cluster(candidates, origin):
    min = 100000
    id = -1
    for idx, neuron in enumerate(candidates[1]):
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            id = idx
    return candidates[1][id]

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def route_distance(cities):
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)

def get_cluster_for_winner(clusters, winner):
    for key, value in clusters.items():
        for neuron in value:
            if np.array_equal(neuron, winner):
                return key


def get_route_m(cities, network, clusters):
   cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest_for_c(network, c),
        axis=1, raw=True)

   cities['cluster'] = cities[['winner']].apply(
        lambda c: get_cluster_for_winner_idx(clusters, network[c]),
        axis=1, raw=True)

    # add depot to each city
   depot = cities.loc[cities['city'] == 'depot']
   depot_skip = depot['cluster'].iloc[0]
   clusters_number = len(clusters)
   for i in range(clusters_number):
       if depot_skip == i:
           depot['winner'] = select_closest_in_cluster(clusters[i], depot[['x', 'y']], network)
           depot.iloc[0, depot.columns.get_loc('cluster')] = i
           continue
       temp = depot.copy(deep=False)
       temp.iloc[0, temp.columns.get_loc('cluster')] = i
       temp.iloc[0, temp.columns.get_loc('winner')] = select_closest_in_cluster(clusters[i], temp[['x', 'y']], network)
       cities = cities.append(temp)

   plot_route_m(cities, 'C:/Users/Mateusz/PycharmProjects/som-tsp/diagrams/route.png')
   return cities

def get_cluster_for_winner_idx(clusters, winner):
    for key, value in clusters.items():
        for neuron in value:
            if np.array_equal(neuron, winner[0]):
                return key