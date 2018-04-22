import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_network_m(cities, neurons_cluster, name='diagram.png', ax=None):
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    colors = ['#0063ba', '#ccff33', '#ff6699', '#660033', '#996633', '#99ff33', '#00ffff', '#006600']
    i = 0
    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)

        for key, value in neurons_cluster.items():
            axis.plot(np.asarray(value)[:,0], np.asarray(value)[:,1], 'r.', ls='-', color=colors[i], markersize=2)
            i = i + 1

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        ax.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)
        return ax

def plot_route(cities, route, name='diagram.png', ax=None):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route['x'], route['y'], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax

def plot_route_m(cities, name='diagram.png', ax=None):
    mpl.rcParams['agg.path.chunksize'] = 10000
    colors = ['#0063ba', '#ccff33', '#ff6699', '#660033', '#996633', '#99ff33', '#00ffff', '#006600']
    i = 0
    fig = plt.figure(figsize=(5, 5), frameon = False)
    axis = fig.add_axes([0, 0, 1, 1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    axis.scatter(cities['x'], cities['y'], color='red', s=4)
    for cluster in cities.groupby('cluster'):
        sorted = cluster[1].sort_values('winner')
        sorted.loc[sorted.shape[0]] = sorted.iloc[0]
        axis.plot(sorted['x'], sorted['y'], color=colors[i], linewidth=1)
        i += 1

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
