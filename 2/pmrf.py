#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy.random as sample
import scipy.stats as pdf
from scipy.io import loadmat
import networkx as nx
import l1l2py
import scipy.optimize
import sympy
import itertools

from sam.sam import *
from factorgraph import *
from sumproduct import *

"""
learn
UGM with binary nodes with only pairwise factors
from full obs



"""

runA = 1
runC = 1

if not __name__ == '__main__': # for import
    runA = 0
    runC = 0

test = 0
save = 0
figs  = 0

def scat(y,**kwargs):
    x = range(len(y))
    plt.scatter(x,y,**kwargs)

"""
senatorVotes (13, 486)
senatorName  (13, 1)
senatorState (13, 1)
senatorParty (13, 1)

N = 13 = |senators|
D = 486 + 3 = |bills| + |features|

name wont matter
state and party will matter

x[n,d] in {0,1}
t[d] in [0,inf)


"""

data = loadmat('data/senate112small.mat')
N = 13
L = 486

D = 13
N = 486

votes = data['senatorVotes'].transpose()
names = data['senatorName']
state = data['senatorState']
party = data['senatorParty']
assert votes.shape == (N,D)

X = votes

""" 2a
algebraic learning on empty pmrf

theta = array([ 0.61522634,  0.70164609,  0.66460905,  0.60493827,  0.58641975,  0.64814815, 0.64814815,  0.61316872,  0.67901235,  0.66666667,  0.66666667,  0.45884774, 0.66255144])

"""
div('2A')
if runA:
    theta = mean(votes, axis=votes.shape.index(L))
    
    if figs:
        figure()
        scat(theta)
        axis((-1,len(theta),0,1))
        show()
        if save:
            savefig('img/2/mle senate vote.png')

""" 2b
functions for the log-likelihood objective and its jacobian/gradient
"""

def triangular(N): return sum(n for n in r(1,N))
assert triangular(4) == 10

def pairs(X):
    """
    
    
    eg N,D = 4,3

    a b c   =>  ab ac bc
    d e f       de df ef
    g h i       gh gi hi
    j k l       jk jl kl
    """

    N,D = X.shape
    out = zeros((N, triangular(D-1)))

    k = 0
    for i in range(D):
        for j in range(i+1,D):
            out[:,k] = X[:,i] * X[:,j]
            k += 1

    return out


#X = a([[0,1,0],[1,1,1],[1,1,0],[0,1,1]])

def f(theta):
    """
    theta = natural parameters of exponential family
    computes joint log-likelihood of data given params
    for maximizing
    fix data, vary params
    """
    assert nan not in theta, 'problem: nan'

    N,D = X.shape
    assert theta.size == triangular(D)
    node_theta = theta[:D]
    edge_theta = theta[D:]

    
    nodes = zeros((2**D, D))
    space = [[0,1] for _ in range(D)]
    for i,x in enumerate(itertools.product(*space)):
        nodes[i,:] = x
    edges = pairs(nodes)

    Z_linear    = nodes.dot(node_theta)
    Z_quadratic = edges.dot(edge_theta)

    linear    = sum(X       .dot(node_theta))
    quadratic = sum(pairs(X).dot(edge_theta))
    log_Z     = log( sum( exp(Z_linear) * exp(Z_quadratic)))

    nll = -(linear + quadratic - log_Z)

    var('theta', theta)
    var('overflow?', max(nodes.dot(node_theta)))
    var('nll', nll)

    return nll


def df(theta):
    """
    gradient/jacobian
    """

    N,D = X.shape
    assert theta.size == triangular(D)
    node_theta = theta[:D]
    edge_theta = theta[D:]

    J = zeros(theta.shape)

    nodes = zeros((2**D, D))
    space = [[0,1] for _ in range(D)]
    for i,x in enumerate(itertools.product(*space)):
        nodes[i,:] = x
    edges = pairs(nodes)

    line_fac = exp(nodes.dot(node_theta)).reshape((2**D,1))
    quad_fac = exp(edges.dot(edge_theta)).reshape((2**D,1))
    node_fac = nodes #: shape((2**D,D))
    edge_fac = edges #: shape((2**D,D))

    line  = sum(X)
    quad  = sum(pairs(X))
    Z_node = log(sum(node_fac * line_fac * quad_fac, axis=0))
    Z_edge = log(sum(edge_fac * line_fac * quad_fac, axis=0))

    J[:D] = line - N * Z_node
    J[D:] = quad - N * Z_edge
    return -J


""" 2c
iterative learning on fully-connected pmrf
unregularized MLE

"""


class J(ndarray):
    """ Jacobian gradient vector of dimensionality D+D*D
    src http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(cls, xs):
        obj = asarray(xs).view(cls)
        
        D = sympy.Symbol('D', integer=True)
        D = max(sympy.solve(D + (D**2 - D)/2 - len(xs)))
        integer = type(sympy.solve(sympy.var('x')-2)[0])
        assert D and D>0 and type(D)==integer, """len(xs) must
        = |nodes| + |edges| for some connected graph
        = D + (D-1 + ... 1) = D + (D**2 - D)/2 for some integer D"""
        
        obj.D = D

        return obj

    def __call__(self, a,b=None):
        if b is None:
            return self[a]
        else:
            # sum D-a..D
            # ~ backwards triangular suffix
            # orders as (0,0) (0,1) .. (0,D)  (1,1) (1,2) .. (1,D)  .. (D,D)
            # undirected edges -> symmetric -> only keep pair once
            assert b > a
            d = self.D
            return self[sum(sam.r(d-a, d)) + (b-a-1)]

if test:
    j = J([0,1,2,3, 0+1,0+2,0+3, 1+2,1+3, 2+3])
    d = j.D
    print 'j =', j
    print 'd =', d
    for a in range(d):
        print 'j(%d) = %d' % (a,j(a))
        assert j(a) == a
    for a in range(d):   
        for b in range(a+1,d):
            print 'j(%d,%d) = %d' % (a,b,j(a,b))
            assert j(a,b) == a+b



def minimize(f,df, x0):
    """ the function 'f', the derivative 'df', the first guess 'x0'
    'df' is called as 'f' is called
    """

    return scipy.optimize.minimize(
        f, x0, method='BFGS', jac=df, options={'maxiter':100})

div('2C')
if runC:
    theta = concatenate([ theta, zeros(triangular(D-1)) ])
    out = minimize(f,df, theta)
    # out.keys() == ['status', 'success', 'njev', 'nfev', 'fun', 'x', 'message', 'hess', 'jac']

    var('optimization', out)
    theta_MLE = out['x']
    #var('theta MLE', theta_MLE)

