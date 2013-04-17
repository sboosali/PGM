#!/usr/bin/python
from __future__ import division

from numpy import *
import numpy as np
import numpy.linalg as la
from numpy.linalg import inv,pinv,det
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy.random as samples
import scipy.stats as pdfs
from scipy.io import loadmat
import numpy.linalg
import time

from sam.sam import *

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

def inv(A):
    #todo because of this "numpy.linalg.linalg.LinAlgError: Singular matrix"
    return la.inv(A) if len(array(A).shape)>1 else 1/A

def pinv(A):
    return la.pinv(A) if len(array(A).shape)>1 else 1/A

def to_symmetric(A): return (A + tp(A))/2

def nans(shape):
    A = empty(shape)
    A[:] = nan
    return A

# # # # # # # # # # # # # # # # # # # # # # # # 
# A
def kalman_smoother(y, A,Q, C,R, x0,V0, EM=False, verbose=False):
    """
    the RTS kalman smoother exactly solves any linear gaussian state space model
    src Rauch-Tung-Striebel (RTS) smoothing (Ghahramani and Hinton, Ch15 of Jordan's textbook)

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

    _P ~> _S ~> S ~> V ~> _V

    if EM:
    compute expected sufficient statistics needed by the E-step
    multiple subscripts and superscripts ==> single indices
    (keep nans in some indices,
    as mutual recursions for other indices must be consistent,
    just dont use them)
    
    """ 

    if not EM:
        T, p = y.shape
        d, = x0.shape
        print
        print '--- Kalman Smoother ---'
        print 'p =', p
        print 'd =', d
        print 'T =', T

    if EM:
        T, p = y.shape
        T = T-1
        d, = x0.shape
 

    # # # # # # # # # # # #
    # Kalman Filter
    x  = nans((T,d))   # x[t] ~ x_{t|t}
    V  = nans((T,d,d)) # V[t] ~ V^{t}_{t}
    K  = nans((T,d,p))
    _x = nans((T,d))
    _V = nans((T,d,d))

    x[0] = x0
    V[0] = V0
    for t in r(1,T-1): # [eg T=4] _,1,2,3
        # time update
        _x[t] = mul(A, x[t-1]) # x_{t|t} ==> x_{t+1|t}
        _V[t] = mul(A, V[t-1], tp(A)) + Q

        # measurement update
        K[t] = mul( _V[t], tp(C), inv(mul(C, _V[t], tp(C)) + R) )
        err  =  y[t] - mul(C, _x[t])
        x[t] = _x[t] + mul(K[t], err) # x_{t+1|t} ==> x_{t+1|t+1}
        V[t] = _V[t] - mul(K[t], C, _V[t]) # next covar not~> data


    # # # # # # # # # # # #
    # Kalman Smoother
    m  = nans((T,d))   # m[t] ~ x_{t|t-1}
    S  = nans((T,d,d)) # S[t] ~ V^T_{t|t-1}
    J  = nans((T,d,d))

    m[-1] = x[-1]
    S[-1] = V[-1]    
    for t in reversed(r(0,T-2)): # [eg T=4] 0,1,2,_
        J[t] = mul( V[t], tp(A), inv(_V[t+1]) ) # K J ...
        m[t] = x[t] + mul( J[t], m[t+1] - _x[t+1] )
        S[t] = V[t] + mul( J[t], S[t+1] - _V[t+1], tp(J[t]) )

    if verbose:
        var('x',x)
        var('V',V)
        var('_x',_x)
        var('_V',_V)
        var('K',K)
        var('m',m)
        var('S',S)
        var('J',J)

    if not EM: return x,V,m,S


    # # # # # # # # # # # #
    # for EM
    _S = nans((T+1,d,d)) # _S[t] ~ V^{T}_{t,t-1}
    P  = nans((T+1,d,d)) # P[t] ~ P_{t}
    _P = nans((T+1,d,d)) # _P[t] ~ P_{t,t-1}
    _S[T] = mul( (identity(d) - mul(K[T], C)) , A , V[T] )

    #print 'm =', m
    for t in reversed(r(2,T-1)):
        _S[t] = mul(V[t], tp(J[t-1])) + mul(J[t], (_S[t+1] - mul(A,V[t])), tp(J[t-1]))
    for t in reversed(r(2,T)):
        _P[t] = _S[t] + mul(m[t], m[t-1])
    for t in reversed(r(1,T)):
        P [t] =  S[t] + mul(m[t], m[t])

    if verbose:
        var('x',x)
        var('V',V)
        var('_x',_x)
        var('_V',_V)
        var('K',K)
        var('m',m)
        var('S',S)
        var('J',J)
        var('P',P)
        var('_P',_P)
        var('_S',_S)
    
    # m,P,P_ for E step
    # _x,_V for mll (my derivation needs the kalman filter's time (not measurement) update)
    if EM: return m, P, _P,  _x, _V


# # # # # # # # # # # # # # # # # # # # # # # # 
# B

data = loadmat('data/track.mat')
Y = data['Y'].transpose()
T, p = Y.shape

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
V0 = identity(d)


xs, Vs, ms, Ss = kalman_smoother(Y, A,Q, C,R, x0,V0, verbose=1)

light_green = (0,0.8,0)
dark_green  = (0,0.5,0)

positions   = [position for (position,velocity) in ms]
confidences = [2*sqrt(S[0,0]) for S in Ss]
plot(Y)
errorbar(range(T), positions, yerr=confidences, color=dark_green)
print '--- Plot ---'
ion();show();time.sleep(60)

# filtered_positions = [position for (position,velocity) in xs]
# smoothed_positions = [position for (position,velocity) in ms]
# plot(Y)
# plot(filtered_positions, color=light_green)
# plot(smoothed_positions, color=dark_green)
# print '--- Plot ---'
# show();time.sleep(60)


# # # # # # # # # # # # # # # # # # # # # # # # 
# C
def marginal_log_likelihood(y, A,Q,C,R, _x,_P):
    """
    compute the marginal log likelihood
    given the learned model parameters
    and the learned expected sufficient statistics

    C must be invertible (:= square and non-singular)
    eg aint no inverse to a fucking projection

    p(y|params) = ...
    """

    if True: return 0

    y  =  y[1:]
    T,p = y.shape
    # _x = _x[1:]
    # _P = _P[1:]
    _,d = _x.shape
    Cy = mul(inv(C), tp(y))
    CRC = inv(mul(tp(C),inv(R),C))

    Z_swap = (1/2) * (log(det(2*pi*CRC)) - log(det(2*pi*R)))

    covar = CRC+identity(d)
    Z_merge = log(2*pi*det(covar))**(-d/2) - (1/2)*mul(tp(Cy), covar, Cy)

    for t in r(1,T-1):
        mean  = _x[t+1]
        covar = CRC + _P[t+1]
        Z_merge += log(2*pi*det(covar))**(-d/2) - (1/2)*mul(tp(Cy-mean), covar, (Cy-mean))

    return sum(T*Z_swap + Z_merge)

# # # # # # # # # # # # # # # # # # # # # # # # 
# D
"""
test that EM learns the constant velocity model that generated the track data
"""

def EM(y, d=None, A=None,Q=None,C=None,R=None, x1=None,P1=None, I=100, verbose=False):
    """
    expectation maximization learns the parameters of a state space model
    when the model is all linear gaussian, the E-step is the kalman smoother
    the M-step is

    only learn A,Q,R (not C,G,x[0],P[0])

    assert the marginal log-likelihood monotonically increases after each M-step
    
    """
    # dims
    T, p = y.shape
    d = int(d) ; assert d>0
    y = cat(nans((1,p)),y)
    # params
    if A is None: A = identity(d)
    if Q is None: Q = identity(d)
    if C is None: C = ones((p,d))
    if R is None: R = identity(p)
    if x1 is None: x1 = zeros((d,1))
    if P1 is None: P1 = identity(d)
    print '--- EM ---'

    _mll = -inf # marginal log-likelihood
    i = 0
    while i<I:
        i += 1
        # make covariance matrices symmetric
        Q = to_symmetric(Q)
        R = to_symmetric(R)

        # E step
        print '-- E %d --' % i
        x,P,_P, time_x,time_P = kalman_smoother(y, A,Q, C,R, x1,P1, EM=True, verbose= verbose and i==I)

        # print 'x P _P'
        # print x
        # print P
        # print _P

        # assert no worse
        # mll = marginal_log_likelihood(y, A,Q,C,R, time_x,time_P)
        # print '%s >= %s' % (mll, _mll)
        # assert mll >= _mll
        # _mll = mll

        # M step
        print '-- M %d --' % i
        A = sum(_P[t] for t in r(2,T)) * inv(sum(P[t-1] for t in r(2,T)))
        Q = 1/(T-1) * (sum(P[t] for t in r(2,T)) - mul(A, sum(tp(_P[t]) for t in r(2,T))))
        R = 1/T * sum(dot(y[t],y[t]) - mul(C, x[t], y[t]) for t in r(1,T))
        x1 = x[1]
        P1 = P[1] - dot(x[1],x[1])
  
        # print 'A Q R'
        # print A
        # print Q
        # print R
        
    return A,Q,R


# # init EM with identities
# data = loadmat('data/track.mat')
# Y = data['Y'].transpose()[:10]
# T, p = Y.shape
# print 'T =', T
# d = 1#2*p

# A = identity(d)
# Q = identity(d)
# C = a([[1]])#a([[1,0]])
# R = identity(p)
# x1 = zeros(d)
# P1 = identity(d)

# A,Q,R = EM(Y, d=d, A=A,Q=Q,C=C,R=R, x1=x1,P1=P1, I=100)
# _, _, Ms, Ss = kalman_smoother(Y, A,Q, C,R, x1,P1)

# # EM should learn the constant velocity model
# p = 1
# d = 2
# sv = 0.1**2
# sx = 1/3 * sv
# sy = 20

# _A = a([[1,1],
#        [0,1]])
# _Q = a([[sx,0],
#        [0,sv]])
# _C = a([1,0])
# _C.shape=(p,d)
# _R = a(sy)
# _R.shape=(p,p)
# _x1 = a([0, 1])
# _P1 = identity(d)

# print
# print 'learned A'
# print A
# print 'true A'
# print _A
# print
# print 'learned Q'
# print Q
# print 'true Q'
# print _Q
# print
# print 'learned R'
# print R
# print 'true R'
# print _R

# smoothed_positions = [position for (position,velocity) in Ms]
# plot(Y)
# plot(smoothed_positions)
# print '--- Plot ---'
# show();time.sleep(60)


Y = loadmat('data/spiral.mat')['Y'].transpose()
T,p = Y.shape
d = p

A = identity(d)
Q = identity(d)
C = zeros((p,d))
R = identity(p)
x1 = zeros(d)
P1 = identity(d)

A,Q,R = EM(Y, d=d, A=A,Q=Q,C=C,R=R, x1=x1,P1=P1, I=10, verbose=False)
_, _, ms, Ss = kalman_smoother(Y, A,Q, C,R, x1,P1)

var('ms',ms)
plot( Y[:,0], Y[:,1], color='b')
plot(ms[:,0],ms[:,1], color='g')
print '--- Plot ---'
ion();show();time.sleep(10)

# # # # # # # # # # # # # # # # # # # # # # # # 
# E
"""
initialize EM with constant position model
EM should learn a constant velocity model
(isnt circle constant acceleration?)
"""


# # # # # # # # # # # # # # # # # # # # # # # # 
# F

