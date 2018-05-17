import numpy as np

def select_closest_m(candidates, origin, network_inhibit):
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

def select_closest_for_cluster(candidates, origin):
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