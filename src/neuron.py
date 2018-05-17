import numpy as np

from distance import select_closest_for_c, select_closest_in_cluster
from plot import plot_route_m

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