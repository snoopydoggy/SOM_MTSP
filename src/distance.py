import numpy as np

from main import get_cluster_for_winner

def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()

def select_closest_m(candidates, origin, network_inhibit):
    min = 100000
    id = -1
    # city = np.array([origin[]])
    for idx, neuron in enumerate(candidates):
        if network_inhibit[idx]:
            continue
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            id = idx
    return id

def select_closest_and_assign(network, c, clusters, cities_clusters):
    min = 100000
    winner_idx = -1
    for idx, neuron in enumerate(network):
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            winner_idx = idx

    for i in range(3):
        cities_clusters[i] = []
    winner = network[winner_idx]
    cluster_id = get_cluster_for_winner(clusters, winner)
    cities_clusters[cluster_id].append(c)
    return id

def select_closest_for_cluster(candidates, origin):
    min = 100000
    id = -1
    for idx, neuron in enumerate(candidates[1]):
        if np.linalg.norm(neuron-origin) < min:
            min = np.linalg.norm(neuron-origin)
            id = idx
    return candidates[1][id]

def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(a - b, axis=1)

def route_distance(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)
