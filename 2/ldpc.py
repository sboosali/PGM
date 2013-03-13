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


runB = 0
runC = 1
runD = 0



"""


"""


""" 1a
: ldpc code => FGM

factor : predicate

test  P(invalid)==0

"""

def ML(marginals):
    return { X:argmax(pX)  for X,pX in marginals.iteritems() }

def isBitvector(B):
    return type(B)==ndarray and all((B==0) + (B==1))

def ldpc_to_fgm(H, debug=True):
    """ : ldpc code matrix => factor graphical model
    H[i,j] == 1  :=  bit j depends on check i

    
    """
    
    n_facs, n_vars = H.shape

    G = FactorGraph()
    G.graph['ldpc'] = True
    def name(j): return 'b%d' % j

    for j in range(n_vars): G.add_var(name=name(j), d=2)
    #  bits are binary randvars named 'b0'..
    for v in G.vars(): G.add_fac(p=[0.5, 0.5], vars=[v])
    #  before set_bits, assume nothing (uniform prior)
    
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


def prior0(eps): return pd([1-eps, eps])
def prior1(eps): return pd([eps, 1-eps])

def set_bits(G,B,eps):
    B = array(B)

    assert 'ldpc' in G.graph, 'set_bits(G,_,_): G must be a LDPC graph'
    assert isBitvector(B), 'set_bits(_,B,_): B must be bitvector'
    assert 0<= eps <=1, 'set_bits(_,_,eps): eps must be in [0,1]'

    priors = { G.node[fac]['vars'][0]: fac
               for fac in G.facs() if len(G.node[fac]['vars'])==1 }

    for var,bit in zip(G.vars(), B):
        # change val
        G.node[var]['val'] = bit

        # change prior
        fac = priors[var]
        G.node[fac]['pmf'] = prior1(eps) if bit==1 else prior0(eps)


""" 1b
msg = zeros(N)

make binary symmetric channel
"""

def coin(p=0.5): return random.random() < p

def flip(bit): return (bit+1) % 2

def channel(bits, eps=0.01): # eps=0.01 -> P(some bit flips) = 1%
    return array([ flip(bit) if coin(eps) else bit  for bit in bits ])

""" 1c
[maybe change the prior, if input isnt always zero vector]

"""

def hamming_distance(x,y):
    assert isBitvector(x) and isBitvector(y) and x.shape==y.shape
    return sum(abs(x-y))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Test

def valid(G):
    # only higher factors
    checks = [f for f in G.facs()  if len(G.node[f]['vars']) > 1]

    # bits are in the vals
    return [G(f, *[G.node[v]['val'] for v in G.node[f]['vars']])
            for f in checks]

def bitstring(B): return ''.join([str(int(b)) for b in B])

def test(G, eps=None, N=None):
    assert N is not None and eps is not None
    
    var('eps', eps)
    var('sparsity H', sum(H) / (n * 2*n))

    input = zeros(2*n) # 0..0 valid under any parity check
    text('channel outputs this.')
    B = channel(input, eps=eps)
    set_bits(G, B, eps)
    var('B', B)
    
    text('channel got what input?')
    Ps = marginalize_sumprod(G, N=N, verbose=1)
    print Ps

    posterior_is_one = [probs[1] for (bit,probs) in 
                        sorted(Ps.items(), key=fst)]

    # observed channel output
    X = B
    # decoded channel output as most probable input
    fX = [int(p>0.5) for p in posterior_is_one]
    # true channel input
    Y = input

    var('valid output?', all(valid(G)))
    var('decode(output) == true input?', all(fX==Y))
    #var('output', bitstring(fX))
    #var('input', bitstring(Y))

    print
    text('decode(output) versus true input')
    color('green', 'o'); print ':= same'
    color('red', 'x');   print ':= diff'
    for fx,y in zip(fX, Y):
        if fx==y:
            color('green', 'o')
        else:
            color('red', 'x')

    return X,fX,Y, Ps,posterior_is_one
    

    
if __name__=='__main__':
    
    div('1A')

    eps = 0

    H = zeros((3,6), dtype=uint8)
    H[0,:] = [1,0,0,0,0,1]
    H[1,:] = [0,1,0,1,0,1]
    H[2,:] = [1,0,1,0,1,0]
    var('H',H)

    B = [1, 0, 1, 1, 0, 1]
    var('B',B)

    G = ldpc_to_fgm(H)
    set_bits(G,B,eps)

    var('G',G.node)

    checks = [f for f in G.facs()  if len(G.node[f]['vars']) > 1]
    var('checks', checks)

    text("it's a valid codeword")
    ok = valid(G)
    print ok
    assert all(ok)

    text("if you flip any one bit, some parity check fails, and the message becomes invalid")
    for v in G.vars():
        G.node[v]['val'] = flip(G.node[v]['val'])

        ok = valid(G)
        print ok
        assert not all(ok)
        
        
        G.node[v]['val'] = flip(G.node[v]['val'])
        



    H = loadmat('data/ldpc36-128.mat')['H']
    n,_ = H.shape
    G = ldpc_to_fgm(H)
    #  its ok to reuse G, so long as you use set_bits


    div('1B')
    """ 1b
    prob. if any parity check fails, any product with its "probabilty" implodes to zero
    soln. 
    
    """
    if runB:

        X,fX,Y,ms,p1 = test(G, eps=0.05, N=50)

        #plot(p1); show()



    div('1C')
    if runC:
        eps = 0.05
        N = 50
        Y = zeros(2*n)

        aux_y = []
        def aux_f(aux_x, aux_y):
            G,Mu = aux_x.G, aux_x.Mu

            Ps = marginals(Mu,G)
            fX = ML(Ps).values()

            aux_y.append(hamming_distance(Y,fX))
            #  order dont matter, since Y is all zeros
        
        for i in range(10):
            #Ps, aux_y = 
    

    div('1D')
    if runD:
        X,fX,Y,ms,p1 = test(G, eps=0.09, N=50)

    




    div('all tests passed!')

