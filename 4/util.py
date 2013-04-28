#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy.random as sample
import scipy.stats as pdfs

import scipy.misc
import itertools
from math import lgamma

from sam.sam import *

# warning  bernoulli(1)==True  but {0,1}=={False,True}
def bernoulli(p): return random.random() < p

def categorical(p, size=None):
    """ Cat K = Multi 1 K """
    assert 0.99<sum(p)<1.01
    if size is None:
        return list(sample.multinomial(1, p)).index(1)
    elif type(size)==int:
        n = size
        return array([list(sample.multinomial(1, p)).index(1) for _ in xrange(n)])
    elif len(size)==2:
        n,m = size
        return array([[list(sample.multinomial(1, p)).index(1) for _ in xrange(m)] for _ in xrange(n)])
    else:pass
sample.categorical = categorical

def cross(N,M): return itertools.product(xrange(N),xrange(M))

def nans(shape, **kwargs):
    A = empty(shape, **kwargs)
    A[:] = nan
    return A

def log_Beta(alphas):
    """
    the beta function is a product-of-gammas over a gamma-of-sum
    the gamma function generalizes factorials
    tested against wolfram alpha
    """
    #return product(map(gamma,alphas)) / gamma(sum(alphas))
    return sum(lgamma(alpha) for alpha in alphas) - lgamma(sum(alphas))

def bwshow(bool_matrix):
    imshow(bool_matrix, cmap='Greys', interpolation='nearest')

def Ber(x,p): return p if x else 1-p

def show(save=True):
    if save:
        plt.show()
    else:
        ion();plt.show();time.sleep(60)

def C(n,r): return int(0.5+scipy.misc.comb(n,r))

def rand_index(x,x_):
    """
    distance between partitions x and x_ on set y
    : [0,1]

    eg
    y  = {a b c}
    x  = [a b b c c  b]
    x_ = [c a a b b  c]
    both_same = 
    both_diff = 
    """
    N = len(x)
    both_same = sum(x[i]==x[j] and x_[i]==x_[j]
                    for i in range(N) for j in range(i+1,N))
    both_diff = sum(x[i]!=x[j] and x_[i]!=x_[j]
                    for i in range(N) for j in range(i+1,N))

    return (both_same + both_diff) / C(N,2)

#rand_index(['a','b','b','c','c','b'],['c','a','a','b','b','c'])

