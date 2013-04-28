#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdfs

from collections import defaultdict
from scipy.io import loadmat
import argparse

from sam.sam import *
from util import *

parser=argparse.ArgumentParser()
parser.add_argument('-iters', type=int, default=500)
parser.add_argument('-runs', type=int, default=5)
parser.add_argument('-save', type=bool, default=True)

args=parser.parse_args()
iters=args.iters
runs=args.runs
save= args.save

burnin = 0#int((1/5) * iters)
skip = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2C

def gibbs(y, N=None,K=None,alpha=1, iters=500,burnin=0,skip=0):
    """
    gibbs sampling for a stochastic block model
    
    N = |nodes|
    K = |classes|
    N2 = |edges| = N^2 - N = 2 * (N-1 + ... 1)

    in my notation,
    x" means "curr x"
    "x_" means "next x"

   
    pi : [0,1]^K^N
    pi[i] : [0,1]^K

    s : {1..K}^|edges|
    s[ij] : {1..K}

    r : {1..K}^|edges|
    r[ij] : {1..K}

    W : [0,1]^(K^2)
    W[k,l] : [0,1]

    y : {0,1}^|y|
    0 <= |y| <= |edges|
    """
    print '--- Gibbs Sampling ---'
    assert N and K
    N2 = N**2 - N
    probs = nans(iters)

    # Init: sample from priors
    # dirichlet prior
    alpha = ones((N,K),dtype=int)
    pi_ = sample.dirichlet(ones(K), size=N)
    pi = pi_.copy()

    # beta prior
    h,t = ones((K,K)), ones((K,K))
    W_ = sample.beta(1,1, size=(K,K))
    W = W_.copy()

    # categorical prior
    r_ = nans((N,N))
    for (i,j) in cross(N,N):
        if i==j: continue
        r_[i,j] = sample.categorical(pi[i])
    r = r_.copy()

    # categorical prior
    s_ = nans((N,N))
    for (i,j) in cross(N,N):
        if i==j: continue
        s_[i,j] = sample.categorical(pi[j])
    s = s_.copy()

    pis,Ws,rs,ss = [pi],[W],[r],[s]
    # Iter: sample from posteriors
    for it in range(iters):
        if (it+1)%100==0: print '- %d -' % (it+1)

        for i in range(N):
            for k in range(K):
                alpha[i,k] += sum(r[i,:]==k) + sum(s[:,i]==k)
            pi_[i] = sample.dirichlet(alpha[i])

        for (k,l) in cross(K,K):
            h[k,l] += sum((y==1) * (r==k) * (s==l))
            t[k,l] += sum((y==0) * (r==k) * (s==l))
        W_ = sample.beta(h,t, size=(K,K))

        r_ = nans((N,N))
        for (i,j) in cross(N,N):
            if i==j: continue
            l = s[i,j]
            rP = array([pi[i][k] * (W[k,l] if y[i,j] else 1-W[k,l])
                        for k in range(K)])
            rP /= sum(rP)
            r_[i,j] = sample.categorical(rP)

        s_ = nans((N,N))
        for (i,j) in cross(N,N):
            if i==j: continue
            k = r[i,j]
            sP = array([pi[j][l] * (W[k,l] if y[i,j] else 1-W[k,l])
                        for l in range(K)])
            sP /= sum(sP)
            s_[i,j] = sample.categorical(sP)

        pi = pi_.copy()
        W  = W_.copy()
        r  = r_.copy()
        s  = s_.copy()
        
        pis.append(pi)
        Ws.append(W)
        rs.append(r)
        ss.append(s)

        # compute log-probability; should (non-monotonically) increase
        Ber = sum(log(W[r[i,j], s[i,j]] if y[i,j] else 1-W[r[i,j], s[i,j]])
                  for (i,j) in cross(N,N) if i!=j and y[i,j]!=nan)
        Cat = sum(log(pi[i][r[i,j]]) +
                  log(pi[j][s[i,j]])
                  for (i,j) in cross(N,N) if i!=j)
        Dir = sum(sum((alpha[i][k]-1) * log(pi[i][k]) for k in range(K))
                  - log_Beta(alpha[i])
                  for i in range(N))
        probs[it] = Ber + Cat + Dir

    return pis,rs,ss,Ws, probs


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
        pis_,zs_,Ws_, probs_,rands_ = gibbs(y, N=N,K=K, iters=iters)
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
# 2D

N = 30
K = 3
W = nans((K,K))
for (k,l) in cross(K,K):
    W[k,l] = 0.10 if k==l else 0.95
z = array([0]*int(N/K) + [1]*int(N/K) + [2]*int(N/K))
y = nans((N,N))
for (i,j) in cross(N,N):
    if i==j: continue
    y[i,j] = bernoulli(W[z[i],z[j]])

probs = [-inf for _ in range(iters)]
for t in xrange(runs):
    print
    print '=== Run %d ===' % (t+1)
    pis_, rs_, ss_, Ws_, probs_ = gibbs(y, N=N,K=K, iters=iters)
    if mean(probs_[burnin:]) > mean(probs[burnin:]):
        pis, rs, ss, Ws, probs = pis_, rs_, ss_, Ws_, probs_
    plot(probs_) # same fig

show(save)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2E

# data = loadmat('data/sampson_network.mat')
# y = data['graph']
# z = data['monk_factions']
# N,_ = y.shape
# K=3

# probs = [-inf for _ in range(iters)]
# for t in xrange(runs):
#     print
#     print '=== Run %d ===' % (t+1)
#     pis_, rs_, ss_, Ws_, probs_ = gibbs(y, N=N,K=K, iters=iters)
#     if mean(probs_[burnin:]) > mean(probs[burnin:]):
#         pis, rs, ss, Ws, probs = pis_, rs_, ss_, Ws_, probs_
#     plot(probs_) # same fig

# to get z from pi,r,s
# z = max(range(K), key=lambda k: pi[k])

# pis = array(pis)[burnin:]
# pi_mean = sum(pis[_] for _ in range(iters))/iters # pairwise mean
# significant = sum(amax(pi_mean, axis=1) > 0.5) # |plurality is majority|
# print '|plurality is majority| / N = %d/%d' % (significant, N)

# show(save)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2F

def gibbs(y, N=None,K=None,alpha=1, iters=500):
    """
    gibbs sampling for a stochastic block model

    keep s and r separate, but sample together

    N = |nodes|
    K = |classes|
    N2 = |edges| = N^2 - N = 2 * (N-1 + ... 1)

    in my notation,
    x" means "curr x"
    "x_" means "next x"

   
    pi : [0,1]^K^N
    pi[i] : [0,1]^K

    rs : ({1..K} x {1..K})^|edges|
    rs[ij] : ({1..K}, {1..K})

    W : [0,1]^(K^2)
    W[k,l] : [0,1]

    y : {0,1}^|y|
    0 <= |y| <= |edges|
    """
    print '--- Gibbs Sampling ---'
    assert N and K
    N2 = N**2 - N
    probs = nans(iters)
    #del rP, sP

    # Init: sample from priors
    # dirichlet prior
    alpha = ones((N,K),dtype=int)
    pi_ = sample.dirichlet(ones(K), size=N)
    pi = pi_.copy()

    # beta prior
    h,t = ones((K,K)), ones((K,K))
    W_ = sample.beta(1,1, size=(K,K))
    W = W_.copy()

    # categorical prior
    r_ = nans((N,N))
    for (i,j) in cross(N,N):
        if i==j: continue
        r_[i,j] = sample.categorical(pi[i])
    r = r_.copy()

    # categorical prior
    s_ = nans((N,N))
    for (i,j) in cross(N,N):
        if i==j: continue
        s_[i,j] = sample.categorical(pi[j])
    s = s_.copy()

    ii = [(k,l) for k,l in cross(K,K)] # inverse index
    pis,Ws,rs,ss = [pi],[W],[r],[s]
    # Iter: sample from posteriors
    for it in range(iters):
        if (it+1)%100==0: print '- %d -' % (it+1)

        for i in range(N):
            for k in range(K):
                alpha[i,k] += sum(r[i,:]==k) + sum(s[:,i]==k)
            pi_[i] = sample.dirichlet(alpha[i])

        for (k,l) in cross(K,K):
            h[k,l] += sum((y==1) * (r==k) * (s==l))
            t[k,l] += sum((y==0) * (r==k) * (s==l))
        W_ = sample.beta(h,t, size=(K,K))

        r_ = nans((N,N))
        s_ = nans((N,N))
        for (i,j) in cross(N,N):
            if i==j: continue
            rsP = array([pi[i][k] * pi[j][l] * (W[k,l] if y[i,j] else 1-W[k,l])
                         for (k,l) in cross(K,K)])
            rsP /= sum(rsP)            
            r_[i,j], s_[i,j] = ii[ sample.categorical(rsP) ]

        pi = pi_.copy()
        W  = W_.copy()
        r  = r_.copy()
        s  = s_.copy()
        
        pis.append(pi)
        Ws.append(W)
        rs.append(r)
        ss.append(s)

        # compute log-probability; should (non-monotonically) increase
        Ber = sum(log(W[r[i,j], s[i,j]] if y[i,j] else 1-W[r[i,j], s[i,j]])
                  for (i,j) in cross(N,N) if i!=j and y[i,j]!=nan)
        Cat = sum(log(pi[i][r[i,j]]) +
                  log(pi[j][s[i,j]])
                  for (i,j) in cross(N,N) if i!=j)
        Dir = sum(sum((alpha[i][k]-1) * log(pi[i][k]) for k in range(K))
                  - log_Beta(alpha[i])
                  for i in range(N))
        probs[it] = Ber + Cat + Dir

    return pis,rs,ss,Ws, probs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2Ga

# N = 30
# K = 3
# W = nans((K,K))
# for (k,l) in cross(K,K):
#     W[k,l] = 0.10 if k==l else 0.95
# z = array([0]*int(N/K) + [1]*int(N/K) + [2]*int(N/K))
# y = nans((N,N))
# for (i,j) in cross(N,N):
#     if i==j: continue
#     y[i,j] = bernoulli(W[z[i],z[j]])

# probs = [-inf for _ in range(iters)]
# for t in xrange(runs):
#     print
#     print '=== Run %d ===' % (t+1)
#     pis_, rs_, ss_, Ws_, probs_ = gibbs(y, N=N,K=K, iters=iters)
#     if mean(probs_[burnin:]) > mean(probs[burnin:]):
#         pis, rs, ss, Ws, probs = pis_, rs_, ss_, Ws_, probs_
#     plot(probs_) # same fig
# show(save);exit()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2Gb

# data = loadmat('data/sampson_network.mat')
# y = data['graph']
# z = data['monk_factions']
# N,_ = y.shape
# K=3

# #iters,runs = 5,2

# probs = [-inf for _ in range(iters)]
# for t in xrange(runs):
#     print
#     print '=== Run %d ===' % (t+1)
#     pis_, rs_, ss_, Ws_, probs_ = gibbs(y, N=N,K=K, iters=iters)
#     if mean(probs_[burnin:]) > mean(probs[burnin:]):
#         pis, rs, ss, Ws, probs = pis_, rs_, ss_, Ws_, probs_
#     plot(probs_) # same fig

# pis = array(pis)[burnin:]
# pi_mean = sum(pis[_] for _ in range(iters))/iters # pairwise mean
# significant = sum(amax(pi_mean, axis=1) > 0.5) # |plurality is majority|
# print '|plurality is majority| / N = %d/%d' % (significant, N)

# show(save)
