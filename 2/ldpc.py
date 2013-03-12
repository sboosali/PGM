#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf
from scipy.io import loadmat
import networkx as nx

from sam.sam import *
from factorgraph import *
from sumproduct import *

eps = 0.05
prior = array([1-eps, eps])



"""


"""


""" 1a
: ldpc code => FGM

factor : predicate

test  P(invalid)==0

"""

def ldpc_to_fgm(H, debug=True):
    """ : ldpc code matrix => factor graphical model
    H[i,j] == 1  :=  bit j depends on check i

    
    """
    
    n_facs, n_vars = H.shape

    G = FactorGraph()
    def name(j): return 'b%d' % j

    for j in range(n_vars): G.add_var(name=name(j), d=2)
    for v in G.vars(): G.add_fac(p=prior, vars=[v])
    
    for i,row in enumerate(H):
        js = [j for j,active in enumerate(row) if active]
        vars = [name(j) for j in js]

        fac = lambda *vals: sum(vals) %2 ==0
        fac.__name__ = 'f%s' % (str(js).replace(' ',''))

        G.add_fac(p=fac, vars=vars, name=fac.__name__)

    if debug:
        alert(':ldpc => :fgm')    
        # var('facs', G.facs())
        # var('vars', G.vars())

    return G


def set_bits(G,B):
    for v,b in zip(G.vars(), B):
        G.node[v]['val'] = b



""" 1b
msg = zeros(N)

make binary symmetric channel
"""

def coin(p=0.5): return random.random() < p

def flip(bit): return (bit+1) % 2

def channel(bits, eps=0.5):
    return array([ bit if coin(eps) else flip(bit)  for bit in bits ])



""" 1c
[maybe change the prior, if input isnt always zero vector]

"""


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Test

if __name__=='__main__': 
    H = zeros((3,6), dtype=uint8)
    H[0,:] = [1,0,0,0,0,1]
    H[1,:] = [0,1,0,1,0,1]
    H[2,:] = [1,0,1,0,1,0]
    var('H',H)

    B = [1, 0, 1, 1, 0, 1]
    var('B',B)

    G = ldpc_to_fgm(H)
    
    set_bits(G,B)

    var('G',G.node)

    checks = [f for f in G.facs()  if len(G.node[f]['vars']) > 1]
    var('checks', checks)
    valid = [ G(f,1,2,3) for f in checks]
    assert all(valid)


    
    GH = loadmat('data/ldpc36-128.mat')
    H = GH['H']
    n,_ = H.shape
    G = ldpc_to_fgm(H)
    
    set_bits(G, channel(zeros(2*n), eps=0.05))
    # var('vars', G.vars())
    # var('facs', G.facs())

    marginals = marginalize_sumprod(G, N=2, verbose=1)
    print marginals
    

    div('all tests passed!')
