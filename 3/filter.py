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

# def show(y):
#     if ioff:
#         plot(y)
#         plt.show()
#         plt.show()
#         time.sleep(60)
#     else:
#         todo

def tp(A): return A.transpose()

def cat(*As): return concatenate(tuple(As))

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

# # # # # # # # # # # # # # # # # # # # # # # # 
# A
def kalman_filter(y, A,G,Q, C,R, x0,P0):
    """
    a kalman filter exactly solves any linear gaussian state space model

    y , data
    A , transition matrix : known
    G , state-noise matrix : known
    Q , state-noise covariance : known
    C , emission matrix : known
    R , data-noise covariance : known
    x0 , init mean
    P0 , init covar

    model is linear-gaussian SSM
    model x[t] in R^d
    model y[t] in R^p
    d , state space dimension

    output x , posterior means estimates : d x T
    output P , posterior covars estimates : d x d x T
    """ 

    T, p = y.shape
    d, = x0.shape
    pinv = numpy.linalg.pinv if p>1 else lambda a: 1/a

    print
    print '--- Kalman Filter ---'
    print 'p =', p
    print 'd =', d
    print 'T =', T

    y = cat(a([[0]]),y)
    x = zeros((T+1,d))
    P = zeros((T+1,d,d))

    x[0] = x0
    P[0] = P0

    for t in range(T):
        # time update
        x[t+1] = mul(A, x[t])
        P[t+1] = mul(A, P[t], tp(A)) + mul(G, Q, tp(G))

        # measurement update
        K      = mul( P[t+1], tp(C), pinv(mul(C, P[t+1], tp(C)) + R) )
        err    = y[t+1] - mul(C, x[t+1])
        x[t+1] = x[t+1] + mul(K, err)
        P[t+1] = P[t+1] - mul(K, C, P[t+1])

    return x[1:], P[1:]

# # # # # # # # # # # # # # # # # # # # # # # # 
# B

"""
constant velocity model
right model
"""

# data = loadmat('data/track.mat')
# Y = data['Y'].transpose()
# T, p = Y.shape

# p = 1
# d = 2
# sv = 0.1**2
# sx = 1/3 * sv
# sy = 20

# A = a([[1,1],
#        [0,1]])
# G = identity(d)
# Q = a([[sx,0],
#        [0,sv]])
# C = a([1,0])
# C.shape=(p,d)
# R = a(sy)
# R.shape=(p,p)
# x0 = a([0, 1])
# P0 = identity(d)


# Xs, Ps = kalman_filter(Y, A,G,Q, C,R, x0,P0)
# positions = [position for (position,velocity) in Xs]
# kalman_positions = positions
# confidences = [2*sqrt(P[0,0]) for P in Ps]

# plot(Y)
# errorbar(range(T), positions, yerr=confidences)
# show();time.sleep(3600)


# # # # # # # # # # # # # # # # # # # # # # # # 
# C

"""
constant position model
wrong model
"""

# data = loadmat('data/track.mat')
# Y = data['Y'].transpose()
# T, p = Y.shape

# p = 1
# d = 1
# sx = 1/3 * 0.1**2
# sy = 20

# A = identity(1)
# Q = a(sx)
# Q.shape = (d,d)
# R = a(sy)
# R.shape = (p,p)
# C = a(1)
# C.shape = (p,d)
# G = identity(d)
# x0 = a([0])
# P0 = identity(d)


# Xs, Ps = kalman_filter(Y, A,G,Q, C,R, x0,P0)
# positions = Xs.reshape(T)
# confidences = [2*sqrt(P) for P in Ps]

# plot(Y)
# errorbar(range(T), positions, yerr=confidences)
# show();time.sleep(3600)

# # # # # # # # # # # # # # # # # # # # # # # # 
# D

def multinomial(xs, ps):
    return xs[ argmax(samples.multinomial(1, ps)) ]


def particle_filter(Y, initial, transition, emission, L):
    """
    a simple bootstrap particle filter approximately solves any state space model
    
    for each moment
    sample from p(x[t] | x[t-1])
    weight by p(y[t] | x[t])
    resample from multinomial(particles, weights)

    y , data

    initial
    = p(x[0])
    can sample
    : () => X

    transition
    = p(x[t] | x[t-1])
    can sample
    : X => X

    emission
    = p(y[t] | x[t])
    can eval
    : Y, X => [0,1]

    L = |particles|

    output X
    T lists of L particles with d dimensions
    """

    T, p = Y.shape
    d,   = initial().shape
    means = empty(T)
    X = [] # : [X]   <-  markov, only need prev particles

    print '--- Particle Filter ---'
    print 'p =', p
    print 'd =', d
    print 'L =', L

    particles = [initial() for l in range(L)]
    weights = [float(emission(Y[0], particles[l])) for l in range(L)]
    weights /= sum(weights)
    X = [multinomial(particles, weights) for l in range(L)]

    for t in range(1,T):
        # sample from p(x[t] | x[t-1])
        particles = a([transition(X[l]) for l in range(L)])

        # weight by p(y[t] | x[t])
        weights = a([float(emission(Y[t], particles[l])) for l in range(L)])
        weights /= sum(weights)

        # weighted mean before resampling (resampling preserves mean, so why? less variance?)
        positions = particles[:,0]
        means[t] = dot(positions, weights)
        
        # resample from multinomial(particles, weights)
        X = [multinomial(particles, weights) for l in range(L)]

    return None, means[1:]


# # # # # # # # # # # # # # # # # # # # # # # # 
# E

data = loadmat('data/track.mat')
Y = data['Y'].transpose()


p = 1
d = 2
sv = 0.1**2
sx = 1/3 * sv
sy = 20

A = a([[1,1],
       [0,1]])
G = identity(d)
Q = a([[sx,0],
       [0,sv]])
C = a([1,0])
C.shape=(p,d)
R = a(sy)
R.shape=(p,p)
x0 = a([0, 1])
P0 = identity(d)

initial    = lambda:     samples.multivariate_normal( [0,1],    identity(d) )
transition = lambda x:   samples.multivariate_normal( dot(A,x), Q )
emission   = lambda y,x: pdfs.norm.pdf( y,            dot(C,x), R )
L = 100

# K = 5
# plot(Y, color='b')
# for k in range(K):
#     X, positions = particle_filter(Y, initial,transition, emission, L)
#     plot(positions, color=(0.2+0.6*(k/(K)),0,0))
# #plot(kalman_positions, color='g')
# print '--- Plot ---'
# show();time.sleep(3600)

# # # # # # # # # # # # # # # # # # # # # # # # 
# F
"""
more particles
"""

# L = 1000
# K = 5
# plot(Y, color='b')
# for k in range(K):
#     X, positions = particle_filter(Y, initial,transition, emission, L)
#     plot(positions, color=(0.2+0.6*(k/K),0,0))
# plot(kalman_positions, color='g')
# print '--- Plot ---'
# show();time.sleep(3600)

# # # # # # # # # # # # # # # # # # # # # # # # 
# G

"""
robust to outliers

"""

eps = 0.1

Y = data['Y'].transpose()
Y = a([[samples.normal(0,40**2)] if coin(eps) else y for y in Y])
T, p = Y.shape

# kalman filter with non-outlier gaussian emission (wrong model)
p = 1
d = 2
sv = 0.1**2
sx = 1/3 * sv
sy = 20

A = a([[1,1],
       [0,1]])
G = identity(d)
Q = a([[sx,0],
       [0,sv]])
C = a([1,0])
C.shape=(p,d)
R = a(sy)
R.shape=(p,p)
x0 = a([0, 1])
P0 = identity(d)

X, P = kalman_filter(Y, A,G,Q, C,R, x0,P0)
kalman_positions = [position for (position,velocity) in X]


# particle filter with outlier mixture-of-gaussian emission (right model)
p = 1
d = 2
sv = 0.1**2
sx = 1/3 * sv
sy = 20

A = a([[1,1],
       [0,1]])
G = identity(d)
Q = a([[sx,0],
       [0,sv]])
C = a([1,0])
C.shape=(p,d)
R = a(sy)
R.shape=(p,p)
x0 = a([0, 1])
P0 = identity(d)

initial    = lambda:     samples.multivariate_normal( [0,1],    identity(d) )
transition = lambda x:   samples.multivariate_normal( dot(A,x), Q )
emission   = lambda y,x: eps*pdfs.norm.pdf(y,0,40**2) + (1-eps)*pdfs.norm.pdf( y, dot(C,x), R )
L = 1000

X, particle_positions = particle_filter(Y, initial,transition, emission, L)


# compare
print '--- Plot ---'
plot(Y)
plot(kalman_positions)
plot(particle_positions,color='r')
axis((0,T) + (-20,100))
show();time.sleep(3600)


