#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdfs

from collections import defaultdict
from scipy.io import loadmat
import argparse

import sam.sam as sam
from util import *

parser=argparse.ArgumentParser()
parser.add_argument('-iters', type=int, default=500)
parser.add_argument('-runs', type=int, default=5)
parser.add_argument('-save', type=bool, default=True)

args=parser.parse_args()
iters=args.iters
runs=args.runs
save = args.save

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1C

def gibbs(y,
          N=None,K=None,alpha=1,
          iters=500,burnin=0,skip=0,
          truth=None):
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
    assert truth is not None
    N2 = N**2 - N
    probs, rands = zeros(iters), zeros(iters)
    true_z = truth

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
        if (it+1)%100==0: print '- %d -' % (it+1)

        for k in xrange(K): alpha[k] += sum(z==k)
        pi_ = sample.dirichlet(alpha)
        #print alpha

        for (k,l) in cross(K,K):
            Ykl = array([y[i,j] for (i,j) in cross(N,N)
                         if i!=j and y[i,j]!=nan and z[i]==k and z[j]==l])
            h[k,l] += sum(Ykl==1)
            t[k,l] += sum(Ykl==0)
        W_ = sample.beta(h,t, size=(K,K))
        #print h,t

        for i in xrange(N):
            zP = array([pi[zi]
                        * prod([W[zi,z[j]] if y[i,j] else 1-W[zi,z[j]]
                                for j in xrange(N)
                                if j!=i and y[i,j]!=nan])
                        * prod([W[z[j],zi] if y[j,i] else 1-W[z[j],zi]
                                for j in xrange(N)
                                if j!=i and y[j,i]!=nan])
                        for zi in xrange(K)])
            zP /= sum(zP)
            z_[i] = sample.categorical(zP)
        #print z

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

        # compute rand_index by samples
        rands[it] = rand_index(true_z, z)

    return pis,zs,Ws, probs,rands


def estimate_z(zs,K):
    iters,N = zs.shape
    return array([max(range(K), key=lambda k: sum(zs[:,i]==k))
                  for i in range(N)])


def viz(y,pi,z,W,
        N,K,
        iters=10,burnin=0,skip=0):
    assert N and K
    probs_figure = figure().number
    rands_figure = figure().number ; ylim(-0.1, 1.1)
    pis, zs, Ws = [],[],[]
    probs = [-inf for _ in range(iters)]
    pispis, zszs, WsWs, probsprobs, randsrands = [],[],[],[],[]
    for t in xrange(runs):
        print
        print '=== Run %d ===' % (t+1)
        pis_,zs_,Ws_, probs_,rands_ = gibbs(y, N=N,K=K, iters=iters, truth=z)
        pispis.append(pis_)
        zszs.append(zs_)
        WsWs.append(Ws_)
        probsprobs.append(probs_)
        randsrands.append(rands_)
        figure(probs_figure); plot(probs_)
        figure(rands_figure); plot(rands_)
        if mean(probs_[burnin:]) > mean(probs[burnin:]):
            pis, zs, Ws = pis_, zs_, Ws_
    figure(probs_figure) ; ylim(ylim()[0] *1.1, ylim()[1] *0.9)
    zs = array(zs)
    
    fig = figure()
    mixing_figure = fig.number
    for run,samples in enumerate(zszs):
        run+=1
        samples = array(samples)
        ax = fig.add_subplot(1,runs, run)
        ax.imshow(samples, interpolation='nearest') # visualize
        ax.get_xaxis().set_ticks([]) # hide
        ax.get_yaxis().set_ticks([]) # hide
        if run==1: # only label first
            ax.get_yaxis().set_ticks([1, iters])
            ax.get_xaxis().set_ticks([0, N-1])
            ax.set_ylabel('iterations') ; ax.set_xlabel('N')
    
    if pi is not None:
        pi_ = sum(pis[_] for _ in range(burnin,iters))/iters # pairwise mean
        print
        print
        print 'true pi'
        print pi
        print
        print 'gibbs pi'
        print pi_
    
    if W is not None:
        W_ = sum(Ws[_] for _ in range(burnin,iters))/iters # pairwise mean
        print
        print
        print 'true W'
        print W
        print
        print 'gibbs W'
        print W_

    if z is not None:
        z_ = estimate_z(zs[burnin:],K) 
        print
        print
        print 'true z  v  gibbs z'
        print 'rand index = %.2f (mean)' % (rand_index(z,z_))
        print 'rand index = %.2f (last)' % (rand_index(z,zs[-1]))

    show(save);exit()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1D

# # here "x_" means "estimated/learned/inferred x"

# N = 30
# K = 3
# W = nans((K,K))
# for (k,l) in cross(K,K):
#     W[k,l] = 0.10 if k==l else 0.95
# z = array([0]*int(N/K) + [1]*int(N/K) + [2]*int(N/K))
# pi = array([sum(z==k)/N for k in range(K)])
# y = nans((N,N))
# for (i,j) in cross(N,N):
#     if i==j: continue
#     y[i,j] = bernoulli(W[z[i],z[j]])

# viz(y=y,pi=pi,z=z,W=W,
#     N=N,K=K,
#     iters=iters,burnin=int(1/5 * iters),skip=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1E

data = loadmat('data/sampson_network.mat')
y = data['graph']
z = data['monk_factions'].flatten()-1 # numpify
N,_ = y.shape
K=4

viz(y=y,pi='?',z=z,W='?',
    N=N,K=K,
    iters=iters,burnin=int(1/5 * iters),skip=0)
