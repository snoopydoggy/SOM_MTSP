import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_network_m(cities, neurons_cluster, name='diagram.png'):
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    colors = ['#0063ba', '#ccff33', '#ff6699', '#660033', '#996633', '#99ff33', '#00ffff', '#006600']
    i = 0
    fig = plt.figure(figsize=(5, 5), frameon = False)
    axis = fig.add_axes([0,0,1,1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    depot = cities.loc[cities['city'] == 'depot']
    axis.scatter(cities['x'], cities['y'], color='red', s=4)
    axis.scatter(depot['x'], depot['y'], color='black', s=8)

    for key, value in neurons_cluster.items():
        axis.plot(np.asarray(value)[:,0], np.asarray(value)[:,1], 'r.', ls='-', color=colors[i], markersize=2)
        axis.plot([np.asarray(value)[-1][0], np.asarray(value)[0][0]], [np.asarray(value)[-1][1], np.asarray(value)[0][1]],
                  color=colors[i], linewidth=1)
        i = i + 1

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_route(cities, route, name='diagram.png'):
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig = plt.figure(figsize=(5, 5), frameon = False)
    axis = fig.add_axes([0, 0, 1, 1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')

    axis.scatter(cities['x'], cities['y'], color='red', s=4)
    route = cities.reindex(route)
    route.loc[route.shape[0]] = route.iloc[0]
    axis.plot(route['x'], route['y'], color='purple', linewidth=1)

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_route_m(cities, name='diagram.png'):
    mpl.rcParams['agg.path.chunksize'] = 10000
    colors = ['#0063ba', '#ccff33', '#ff6699', '#660033', '#996633', '#99ff33', '#00ffff', '#006600']
    i = 0
    fig = plt.figure(figsize=(5, 5), frameon=False)
    axis = fig.add_axes([0, 0, 1, 1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    depot = cities.loc[cities['city'] == 'depot']
    axis.scatter(cities['x'], cities['y'], color='red', s=4)
    axis.scatter(depot['x'], depot['y'], color='black', s=8)

    for cluster in cities.groupby('cluster'):
        sorted = cluster[1].sort_values('winner')
        axis.plot(sorted['x'], sorted['y'], color=colors[i], linewidth=1)
        axis.plot([sorted.iloc[-1]['x'], sorted.iloc[0]['x']], [sorted.iloc[-1]['y'], sorted.iloc[0]['y']], color=colors[i], linewidth=1)
        i += 1

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_route_sim(cities, name):
    mpl.rcParams['agg.path.chunksize'] = 10000
    colors = ['#0063ba', '#ccff33', '#ff6699', '#660033', '#996633', '#99ff33', '#00ffff', '#006600']
    i = 0
    fig = plt.figure(figsize=(5, 5), frameon=False)
    axis = fig.add_axes([0, 0, 1, 1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    depot = cities.loc[cities['city'] == 'depot']
    axis.scatter(cities['x'], cities['y'], color='red', s=4)
    axis.scatter(depot['x'], depot['y'], color='black', s=8)

    for cluster in cities.groupby('cluster'):
        sorted = cluster[1].sort_values('winner')
        axis.plot(sorted['x'], sorted['y'], color=colors[i], linewidth=1)
        axis.plot([sorted.iloc[-1]['x'], sorted.iloc[0]['x']], [sorted.iloc[-1]['y'], sorted.iloc[0]['y']], color=colors[i], linewidth=1)
        unvisited = sorted.query('visited==True')
        axis.plot(unvisited['x'], unvisited['y'], color='black')
        i += 1

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()