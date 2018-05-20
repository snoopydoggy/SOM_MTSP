from sys import argv

import numpy as np
import math as math

from io_helper import read_mtsp, normalize
from neuron_helper import euclidean_distance, route_distance, select_closest_neuron, select_closest_neuron_for_cluster, \
    get_route
from plot import plot_route, plot_network_m


def main():
    test_data = read_mtsp(argv[1])
    solve_algorithm(test_data, argv[2])


def solve_algorithm(test_data, tsps_number):
    cities = test_data.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']])
    n = cities.shape[0] * 4
    G = 0.4 * n
    alfa = 0.03
    learning_rate = 0.6
    weight = 0.3
    k = int(tsps_number)
    nodes_per_cluster = math.floor(n / k)
    H = 0.2 * nodes_per_cluster
    network_size = nodes_per_cluster * k
    network = np.zeros(shape=(network_size, 2))
    network_inhibit = np.zeros((n,), dtype=bool)
    clusters = generate_networks(n, network, k)

    depot = cities.loc[cities['city'] == 'depot']
    depot = depot[['x', 'y']].values[0]
    for i in range(160):

        for cluster in clusters.items():
            winner = select_closest_neuron_for_cluster(cluster, depot)
            update_network_inhibit_for_winner(winner, network, network_inhibit)
            update_cluster_values(cluster[1], depot, learning_rate, weight, winner, G, H)

        for city in cities[['x', 'y']].values:
            winner_idx = select_closest_neuron(network, city, network_inhibit)
            network_inhibit[winner_idx] = 1
            winner = network[winner_idx]
            cluster_id = get_cluster_for_winner(clusters, winner)
            update_cluster_values(clusters[cluster_id], city, learning_rate, weight, winner, G, H)

        G = G * (1 - alfa)
        learning_rate = learning_rate * (1 - alfa)
        weight = weight * 0.9
        network_inhibit = np.zeros((n,), dtype=bool)

        plot_network_m(cities, clusters, name='C:/Users/Mateusz/PycharmProjects/som-tsp/diagrams/{:05d}.png'.format(i))

    plot_network_m(cities, clusters, name='C:/Users/Mateusz/PycharmProjects/som-tsp/diagrams/final.png')

    total_distance = 0

    cities = get_route(cities, network, clusters)
    cities = cities.sort_values('winner')
    test_data = test_data.reindex(cities.index)
    test_data['cluster'] = cities['cluster']
    for cluster in test_data.groupby('cluster'):
        temp_distance = route_distance(cluster[1])
        print('Distance: {} for salesman: {}'.format(temp_distance, cluster[1]['cluster'][0]))
        total_distance += temp_distance
    print('Total distance: {} '.format(total_distance))
    return


def generate_networks(n, network, k=1):
    r = 0.25
    arc = 360 / k
    currentArc = 0
    centers = []
    clusters = {}
    neurons_per_cluster = math.floor(n / k)
    for i in range(k):
        cix = 0.5 + r * math.cos(math.radians(currentArc))
        ciy = 0.5 + r * math.sin(math.radians(currentArc))
        centers.append([cix, ciy])
        currentArc += arc
        clusters[i] = []
    cluster_radius = np.linalg.norm(np.array(centers[0]) - np.array(centers[1])) / 2

    arc_per_neuron = 360 / neurons_per_cluster
    n_i = 0
    for idx, center in enumerate(centers):
        currentArc = 0
        for i in range(neurons_per_cluster):
            cix = center[0] + cluster_radius * math.cos(math.radians(currentArc))
            ciy = center[1] + cluster_radius * math.sin(math.radians(currentArc))
            network[n_i] = np.array([cix, ciy])
            clusters[idx].append(network[n_i])
            currentArc += arc_per_neuron
            n_i += 1
    return clusters


def get_cluster_for_winner(clusters, winner):
    for key, value in clusters.items():
        for neuron in value:
            if np.array_equal(neuron, winner):
                return key


def update_cluster_values(cluster, city, learning_rate, weight, winner, G, H):
    M = len(cluster)
    for idx, neuron in enumerate(cluster):
        delta0 = neighborhood_function(winner, idx, cluster, G, H, M) * learning_rate * (
                    city[0] - neuron[0]) + weight * (
                             previous(cluster, idx)[0] - (2 * neuron[0]) + next_neuron(cluster, idx)[0])
        delta1 = neighborhood_function(winner, idx, cluster, G, H, M) * learning_rate * (
                    city[1] - neuron[1]) + weight * (
                             previous(cluster, idx)[1] - (2 * neuron[1]) + next_neuron(cluster, idx)[1])
        neuron[0] = neuron[0] + delta0
        neuron[1] = neuron[1] + delta1


def neighborhood_function(winner, idx, cluster, G, H, M):
    # get winner idx
    winner_id = 0
    for j, neuron in enumerate(cluster):
        if np.array_equal(neuron, winner):
            winner_id = j

    temp = math.fabs(idx - winner_id)

    distance = temp if temp < M - temp else M - temp

    if (distance > H):
        return 0

    return math.exp((-distance * distance) / (G * G))


def previous(cluster, idx):
    if idx == 0:
        return cluster[len(cluster) - 1]
    return cluster[idx - 1]


def next_neuron(cluster, idx):
    if idx == len(cluster) - 1:
        return cluster[0]
    return cluster[idx + 1]


def update_network_inhibit_for_winner(winner, network, network_inhibit):
    for idx, neuron in enumerate(network):
        if np.array_equal(neuron, winner):
            network_inhibit[idx] = 1


if __name__ == '__main__':
    main()
