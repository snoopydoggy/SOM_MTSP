from sys import argv

import numpy as np
import math as math

from io_helper import read_tsp, normalize
from neuron import get_route, get_route_m
from distance import select_closest, euclidean_distance, route_distance, select_closest_m, select_closest_for_cluster
from plot import plot_route, plot_network_m

def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1

    problem = read_tsp(argv[1])

    route = som(problem, 100000)

    problem = problem.reindex(route)

    distance = route_distance(problem)

    print('Route found of length {}'.format(distance))


def som(problem, iterations):
    #att48.tsp depot: city = 21
    cities = problem.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']])
    n = cities.shape[0] * 4
    m = n
    G = 0.4 * n
    k = 3
    alfa = 0.03
    learning_rate = 0.6
    weight = 0.3

    nodes_per_cluster = math.floor(n/k)
    H = 0.2 * nodes_per_cluster
    network_size = nodes_per_cluster * k
    network = np.zeros(shape=(network_size, 2))
    network_inhibit = np.zeros((n,), dtype=bool)

    clusters = generateClusters_m(n, network, k)
    temp = 0
    depot = cities.loc[cities['city'] == 'depot']
    depot = depot[['x', 'y']].values[0]
    plot_network_m(cities, clusters,
               name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/before.png')
    for i in range(400):

        for cluster in clusters.items():
            winner = select_closest_for_cluster(cluster, depot)
            update_network_inhibit_for_winner(winner, network, network_inhibit)
            update_cluster_values(cluster[1], depot, learning_rate, weight, winner, G, H)
            #plot_network_m(cities, clusters,
            #           name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/after.png')
        for city in cities[['x', 'y']].values:
            winner_idx = select_closest_m(network, city, network_inhibit)
            network_inhibit[winner_idx] = 1
            winner = network[winner_idx]
            cluster_id = get_cluster_for_winner(clusters, winner)
            update_cluster_values(clusters[cluster_id], city, learning_rate, weight, winner, G, H)
            # plot_network_m(cities, clusters, name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/{:05d}.png'.format(temp))
            temp += 1

        G = G * (1-alfa)
        learning_rate = learning_rate * (1-alfa)
        network_inhibit = np.zeros((m,), dtype=bool)

        weight = weight * 0.9

        plot_network_m(cities, clusters, name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/{:05d}.png'.format(i))


    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network_m(cities, clusters, name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/final.png')

    route = get_route(cities, network)
    get_route_m(cities, network, clusters)
    return route


def generateClusters_m(n, network, k=1):
    r = 0.25
    arc = 360/k
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
    cluster_radius = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))/2

    arc_per_neuron = 360/neurons_per_cluster
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
        delta0 = neighborhood_function(winner, idx, cluster, G, H, M) * learning_rate * (city[0] - neuron[0]) + weight * (previous(cluster, idx)[0] - (2 * neuron[0]) + next_neuron(cluster, idx)[0])
        delta1 = neighborhood_function(winner, idx, cluster, G, H, M) * learning_rate * (city[1] - neuron[1]) + weight * (previous(cluster, idx)[1] - (2 * neuron[1]) + next_neuron(cluster, idx)[1])
        neuron[0] = neuron[0] + delta0
        neuron[1] = neuron[1] + delta1
        # plot_network_m(cities, clusters,
        #                name='C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/10' + str(idx) + '.png')

def neighborhood_function(winner, idx, cluster, G, H, M):
    # get winner idx
    winner_id = 0
    for j, neuron in enumerate(cluster):
        if np.array_equal(neuron, winner):
            winner_id = j

    temp = math.fabs(idx - winner_id)

    distance = temp if temp < M - temp else M - temp

    if(distance > H):
        return 0

    return math.exp((-distance*distance)/(G*G))

def previous(cluster, idx):
    if idx == 0:
        return cluster[len(cluster)-1]
    return cluster[idx-1]

def next_neuron(cluster, idx):
    if idx == len(cluster)-1:
        return cluster[0]
    return cluster[idx+1]

def update_network_inhibit_for_winner(winner, network, network_inhibit):
    for idx, neuron in enumerate(network):
            if np.array_equal(neuron, winner):
                network_inhibit[idx] = 1

if __name__ == '__main__':
    main()


