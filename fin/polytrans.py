#!/usr/bin/python
from __future__ import division
from sam.sam import *

from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy.random as samples
import scipy.stats as pdfs
from scipy.io import loadmat
import numpy.linalg
import time

def cat(*As): return concatenate(As)

def mul(*As):
    """
    left-associate matrix multiplication with helpful dimension mismatch error message
    """
    prod = 1
    for i,A in enumerate(As):
        try:
            prod = dot(prod,A)
        except ValueError as e:
            if e.message == 'matrices are not aligned':
                print
                print 'mul(%s)' % ('B,'*len(As[:i]) + 'A,' + '_,'*len(As[i+1:]))
                print 'A.shape  = %s' % A.shape
                print 'Bs.shape = %s' % prod.shape
                print
            raise e

    return prod

def coin(p=0.5): return random.random() < p

def multinomial(xs, ps):
    return xs[ argmax(samples.multinomial(1, ps)) ]

def nans(shape):
    A = empty(shape)
    A.fill(nan)
    return A

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def particle_filter(Y, initial, sample, weigh, L):
    """
    a simple bootstrap particle filter approximately solves any state space model
    
    for each moment
    particles <== sample _
    weights <== weigh _ _
    resample from multinomial(particles, weights)

    y , data

    initial
    can sample
    : => X

    sample
    eg p(x[t] | x[t-1])
    can sample
    : _ => _

    weigh
    eg p(y[t] | x[t])
    can eval
    : _, _ => [0,1]

    L = |particles|

    output X
    T lists of L particles with d dimensions
    X[t,l,d]
    """

    T, p = Y.shape
    d = initial().shape

    print '--- Particle Filter ---'
    print 'p =', p
    print 'd =', d
    print 'L =', L

    X = [initial() for _ in range(L)]

    for y in Y:
        # sample
        particles = a([sample(y) for _ in range(L)])

        # weigh
        weights = a([weigh(X, particle) for particle in particles])
        weights /= sum(weights)
      
        # resample
        X = [multinomial(particles, weights) for _ in range(L)]

        yield X

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Test

