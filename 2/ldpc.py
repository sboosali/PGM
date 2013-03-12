#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf
from scipy.io import loadmat as load
import networkx as nx

from sam.sam import *
from factorgraph import *
from sumprod import *

GH = load('ldpc36-128.mat')
G,H = GH['G'], GH['H']
N,_ = H.shape

eps = 0.05
iters = 50
prior = array([1-eps, eps])



"""


"""


""" 1a
: ldpc code => FGM

factor : predicate

test  P(invalid)==0

"""

# : ldpc code matrix => factor graphical model
def ldpc_to_fgm(H):
    G = FactorGraph()
    for _ in range(n):
        G.add_var()


""" 1b
msg = zeros(N)

make binary symmetric channel
"""

def coin(p=0.5): return random.random() < p

def flip(bit): return (bit+1) % 2

def channel(bits, eps=0.05):
    return array([ bit if coin(eps) else flip(bit)  for bit in bits ])

