import numpy as np

from distance import select_closest, select_closest_for_c
from plot import plot_route_m

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(cities, network):
    """Return the route computed by a network."""
    cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)

    return cities.sort_values('winner').index

def get_route_m(cities, network, clusters):
   cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest_for_c(network, c),
        axis=1, raw=True)

   cities['cluster'] = cities[['winner']].apply(
        lambda c: get_cluster_for_winner_idx(clusters, network[c]),
        axis=1, raw=True)

   plot_route_m(cities, 'C:/Users/Mateusz/PycharmProjects/som-tsp/tempdiagrams/route.png')


def get_cluster_for_winner_idx(clusters, winner):
    for key, value in clusters.items():
        for neuron in value:
            if np.array_equal(neuron, winner[0]):
                return key