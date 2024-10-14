import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import linalg as la
import math
import scipy
from tabulate import tabulate
from matplotlib.colors import ListedColormap

def update(x, rule, n):
    next = [0]*n
    for i in range(n):
        next[i] = rule[(x[(i - 1) % n],x[i],x[(i + 1) % n])]
    return next
 
def pOne():
    """4 Cell Lattice with periodic bdy conditions"""
    N = 4
    rule = {
            (1,1,1): 0,
            (1,1,0): 1,
            (1,0,1): 1,
            (1,0,0): 0,
            (0,1,1): 1,
            (0,1,0): 1,
            (0,0,1): 1,
            (0,0,0): 0
            }
    g = nx.DiGraph()
    for v in range(2**N):
        # Creates lattice elements from int v
        x = [1 if v & 2**i > 0 else 0 for i in range(N - 1, -1, -1)]
        # end node is created by converting state to int
        g.add_edge(v, int(''.join(map(str, update(x, rule, N))), 2))
    print(g)
    connected_components = [c for c in nx.connected_components(g.to_undirected())]
    w = math.ceil(math.sqrt(len(connected_components)))
    h = math.ceil(len(connected_components) / w)

    plt.figure(1, figsize = (12,12))
    # plt.figure(1, figsize = (20,20))
    for i in range(len(connected_components)):
        plt.subplot(h, w, i + 1)
        nx.draw_planar(nx.subgraph(g, connected_components[i]), with_labels = True)

    # plt.show()
    # print(nx.adjacency_matrix(g))
    adj = nx.to_numpy_array(g)
    # NOTE: Need to change 04b when updating N
    headers = [f'{x:04b}' for x in range(2**N)]
    with open('output/qOneAdj.txt', 'w') as f:
        print(tabulate(adj, headers, tablefmt="latex"), file=f)

    plt.savefig(f'output/qOne.png')
    plt.close()
