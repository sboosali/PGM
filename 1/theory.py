#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import networkx as nx
import time
import itertools
from copy import deepcopy

from sam.sam import *

def make_grid():
    grid = nx.grid_graph([3,3])
    grid = nx.relabel_nodes(grid, {
            (0,0): 9,
            (1,0): 6,
            (2,0): 3,
            (0,1): 8,
            (1,1): 5,
            (2,1): 2,
            (0,2): 7,
            (1,2): 4,
            (2,2): 1,
            })
    return grid

def cliquify(G, xs):
    # xs = list(xs)
    # for i,x in enumerate(xs):
    #     for y in xs[i+1:]:
    #         G.add_edge(x,y)
    for x in xs:
        for y in xs:
            if y==x: continue
            G.add_edge(x,y)

def elim(G, x):
    cliquify(G, G.edge[x].keys())
    G.remove_node(x)

def elims(G, elimination_ordering): 
    for x in elimination_ordering: elim(G,x)

def treewidth(graph):
    """
    treewidth = min max-elimination-clique
    """

    print 'treewidth( %s )' % graph
    
    tw = +inf
    G = deepcopy(graph)
    nodes = deepcopy(graph.node.keys())

    _G = deepcopy(G)
    _tw = 0
    for elimination_ordering in itertools.permutations(nodes):
        
        for node in elimination_ordering:
            elim(_G, node)
            if _G: _tw = max(_tw, nx.graph_clique_number(_G))

        tw = min(_tw, tw)
        _G = deepcopy(G)
        _tw = 0

    return tw

# G = make_grid()
# elimination_ordering = [5,2,6,8, 4,9,7,3,1]
# for i,x in enumerate(elimination_ordering):
#     # figure();nx.draw(G);savefig('A%d.png' % (i+1))
#     elim(G,x)

# G = make_grid()
# elimination_ordering = [9,7,3,6, 8,5,2,4,1]
# start = time.clock()
# for i,x in enumerate(elimination_ordering):
#     # figure();nx.draw(G);savefig('B%d.png' % (i+1))
#     elim(G,x)
# end = time.clock()

# _elims = timeit(elims)
# G = make_grid()
# _,t = _elims(G, [9,7,3,6, 8,5,2,4,1])
# print t * 1e8 / (60*60)


"""

a 2x2 grid has 4! elimination orderings and takes milliseconds
a 3x3 grid has 9! elimination orderings and takes minutes

"""

_treewidth = timeit(treewidth)
G = nx.grid_graph([3,3])
tw,t = _treewidth(G)
var('time', t)
var('treewidth', tw)

