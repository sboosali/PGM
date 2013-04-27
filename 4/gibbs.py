#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdfs

from collections import defaultdict

from sam.sam import *
from util import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1C

def gibbs(y, N=None,K=None,alpha=1, iters=500, burnin=0, skip=0):
    """
    gibbs sampling for a stochastic block model
    
    N = |nodes|
    K = |classes|
    N2 = |edges| = N^2 - N = 2 * (N-1 + ... 1)

    in my notation,
    x" means "curr x"
    "x_" means "next x"

    pi : [0,1]^K
    pi[k] : [0,1]

    z : {1..K}^N
    z[i] : {1..K}

    W : [0,1]^(K^2)
    W[k,l] : [0,1]

    y : {0,1}^|y|
    0 <= |y| <= N2
    """

    print '--- Gibbs Sampling ---'
    assert N and K
    assert 0 <= burnin < iters and 0 <= skip < iters
    N2 = N**2 - N
    probs = nans(iters)

    # Init: sample from priors
    # dirichlet prior
    alpha = ones(K,dtype=int)
    pi_ = sample.dirichlet(ones(K))
    pi = pi_.copy()

    # beta prior
    h,t = ones((K,K)), ones((K,K))
    W_ = sample.beta(1,1, size=(K,K))
    W = W_.copy()

    # categorical prior
    z_ = nans(N,dtype=int)
    for i in xrange(N):
        z_[i] = sample.categorical(pi)
    z = z_.copy()

    pis, Ws, zs = [pi],[W],[z]
    # Iter: sample from posteriors
    for it in xrange(iters):
        print '- %d -' % (it+1)        

        for k in xrange(K):
            alpha[k] += sum(z==k)
        pi_ = sample.dirichlet(alpha)
        #print alpha

        for (k,l) in cross(K,K):
            Ykl = array([y[i,j] for (i,j) in cross(N,N)
                         if i!=j and y[i,j]!=nan and z[i]==k and z[j]==l])
            h[k,l] += sum(Ykl==0)
            t[k,l] += sum(Ykl==1)
        W_ = sample.beta(h,t, size=(K,K))

        #print z
        for i in xrange(N):
            zP = array([pi[k]
                        * prod([W[k,z[j]] if y[i,j] else 1-W[k,z[j]]
                                for j in xrange(N)
                                if j!=i and y[i,j]!=nan])
                        * prod([W[z[j],k] if y[j,i] else 1-W[z[j],k]
                                for j in xrange(N)
                                if j!=i and y[j,i]!=nan])
                        for k in xrange(K)]) # z[i]=k
            zP /= sum(zP)
            z_[i] = sample.categorical(zP)

        pi = pi_.copy()
        W  = W_.copy()
        z  = z_.copy()

        pis.append(pi)
        Ws.append(W)
        zs.append(z)
        # compute log-probability; should (non-monotonically) increase
        Ber = sum(log(W[z[i],z[j]] if y[i,j] else 1-W[z[i],z[j]])
                  for (i,j) in cross(N,N) if i!=j and y[i,j]!=nan)
        Cat = sum(log(pi[z[i]]) for i in range(N))
        Dir = sum((alpha[k]-1) * log(pi[k])
                  for k in xrange(K)) - log_Beta(alpha)
        probs[it] = Ber + Cat + Dir

    return pis,zs,Ws, probs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2D

N = 30
K = 3
W = nans((K,K))
for (k,l) in cross(K,K):
    W[k,l] = 0.10 if k==l else 0.95
#z = numpy.random.permutation([0]*int(N/K) + [1]*int(N/K) + [2]*int(N/K))
z = array([0]*int(N/K) + [1]*int(N/K) + [2]*int(N/K))
pi = array([sum(z==k)/N for k in range(K)])
y = nans((N,N))
for (i,j) in cross(N,N):
    if i==j: continue
    y[i,j] = bernoulli(W[z[i],z[j]])

# here "x_" means "estimated/learned/inferred x"
iters = 500
runs = 1
burnin = int(iters*4/5)
skip = 0

pis, zs, Ws = [],[],[]
probs = [-inf for _ in range(iters)]
for t in xrange(runs):
    print
    print '=== Run %d ===' % t
    pis_, zs_, Ws_, probs_ = gibbs(y, N=N,K=K, iters=iters)
    if mean(probs_[burnin:]) > mean(probs[burnin:]):
        pis, zs, Ws, probs = pis_, zs_, Ws_, probs_
    plot(probs) # same fig

pi_ = sum(pis[_] for _ in range(iters))/iters
W_ = sum(Ws[_] for _ in range(iters))/iters
zs = array(zs)
z_ = array([max(range(K), key=lambda k: sum(zs[burnin:,i]==k))
            for i in range(N)])

print
print
print 'true z'
print z
print
print 'gibbs z'
print z_
print
print 'true z == gibbs z'
print '(%.3f is chance)' % (1/K)
print '%d / %d' % (sum(z==z_), N)

print
print
print 'true pi'
print pi
print
print 'gibbs pi'
print pi_

print
print
print 'true W'
print W
print
print 'gibbs W'
print W_

ion();show();time.sleep(60)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1D

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1E
