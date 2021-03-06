#!/usr/bin/python -W ignore::Warning
from __future__ import division
from numpy import *
from matplotlib.pyplot import *
import numpy.random as samples
import scipy.stats as pdfs

import argparse
import time
from itertools import groupby

from sam import sam
from util import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Infer

def particle_filter(Y, X0, sample, weigh, L=100, K=3, gamma=1):
    """
    a simple bootstrap particle filter approximately solves any state space model
    samples from likelihood and weighs by transition
    Y is very informative
    X is just for smoothness, sparsity, noise
    
    for each moment
    particles <== sample _
    weights <== weigh _ _
    resample from multinomial(particles, weights)

    K=1 and gamma=1 recovers old model

    input Y , data : stream

    input X0
    can sample
    : => X

    input sample
    eg p(y[t] | x[t])
    can sample
    : _ => _

    input weigh
    eg p(x[t] | x[t-1])
    can eval
    : _, _ => [0,1]

    input L = |particles|

    input K = |markov order|

    input gamma = discounting for higher-order markov chain

    output X
    T lists of L particles with d dimensions
    X[t,l,d]

    """
    print '--- Particle Filter (L=%d) ---' % L

    XXX = [a([X0() for _ in range(L)])] # : (_,L,D)
    d = len(X0())
    ymax = 1e6

    for t,y in enumerate(Y):
        print '- %d -' % (t+1)
        ymax = max(ymax, y.max())

        # sample
        particles = sample(y, L, ymin=0, ymax=ymax)

        weights = weigh(reversed(XXX[:-(K+1):-1]), particles, gamma=gamma)
        weights /= sum(weights)

        # resample
        XX = a([multinomial(particles, weights) for _ in range(L)])
        XXX.append(XX)

        # infer
        X = [max(var, key=lambda val: sum(val==XX[:,_])) for _ in range(d)]
        yield X


def fft_infer(Y):
    for y in Y:
        _freqs = {i*window_rate : y[i] for i in range(len(y))} # y[fft basis] => y[freq basis]
        _notes = {n : max(_freqs[f] for f in fs) # max ampl of freqs near note
                  for n,fs in groupby(sorted(_freqs),key=note)} # group freqs by note
        yield a([_notes.get(notes[j],0) for j in range(d)])

def nmf(A, B, iters=50, verbose=True):
    """
    jointly solve AX=B for X where X>=0
    multiplicative update with euclidean distance
    """
    assert all(B>=0)
    assert all(A>=0)
    if B.ndim==1: B.shape = 1, B.size
    T, p = B.shape
    d, p = A.shape
    A = A.T
    B = (B / sum(B)).T
    X = (1/d) * ones((d, T))

    for i in range(iters):
        if verbose: print '%d/%d' % (i+1,iters)
        X = X * mul( A.T, B ) / mul( A.T, A, X )

    assert all(X>=0)
    if T==1: X.shape = (d,)
    return X

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model ~> Input

cmd=argparse.ArgumentParser(description='Polyphonic Transcription by Particle Filter with Likelihood Samples')
cmd.add_argument('file', help='a .wav audio file, the input to transcribe')
cmd.add_argument('base', help='a dir of .wav audio files, defines what notes are')
cmd.add_argument('-L', type=int, default=100,
                 help='the number of particles in the particle filter')
cmd.add_argument('-K', type=int, default=1,
                 help='the order of the markov chain (in windows)')
cmd.add_argument('-gamma', type=float, default=1,
                 help='the discount factor for values of past states (when K>1)')
inference_algorithms = ['pf', 'fft', 'nmf']
cmd.add_argument('-by', 
                 type=lambda x: x if x in inference_algorithms else None, default='pf',
                 help='one of [pf fft nmf], the inference algorithm')
args=cmd.parse_args()

window_size = 2**12 # 44100 samples/second / 2^12 samples/window = 10 windows/second
A,freqs,notes,sample_rate = basis(args.base, truncate=44100*5, window_size=window_size)


# the graphical model is a factoiral HMM
var = [0,1]
d,_ = A.shape

def poi(x,mu): return pdfs.poisson.pmf(x,mu)
#TODO data-driven
p00 = 0.01 # P(silence sticks)
p11 = 0.999 # P(sound sticks)
transition = a([[p00, 1-p00],
                [1-p11, p11]])
def weigh(XXX, particles, gamma=0.9):
    """
    XXX : (K,L,D)
    XXX[0] ~ X[t-1]
    XXX[K-1] ~ X[t-K]
    particles : (L,D)
    """
    #TODO check {gamma^k ..} is not backwards
    W = a([ sum( gamma**(k+1) * poi(sum(X),1) * product([transition[prev,curr] for (prev, curr) in zip(XX[l], particle)])
                 for (k, XX) in enumerate(XXX))
            for (l, particle) in enumerate(particles)])
    return W

def sample(y, L, ymin=None, ymax=None):
    """
    : audio => note

    min/max energies
    changes greatly given different audio, format, recording, processing, etc
    smoothly between . absolute metric for low energies . relative metric for high energies

    y[i]
    fft is wrt cycles / samples/window  but  freqs are wrt cycles / second
    (cycles / samples/window) * (1 / windows/second) * (samples / seconds) = cycles / second = Hz

    """
    if not ymin: ymin = y.min()
    if not ymax: ymax = y.max()

    x = nmf(A, y/ymax, iters=50, verbose=False)
    return a([[ber(x[j]) for j in range(d)]
              for _ in range(L)])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Main

Y, sample_rate = fft_wav(args.file, window_size=window_size)
T, _ = Y.shape # time in windows
window_rate = sample_rate / window_size # samples/second / samples/window = windows/second


if args.by=='fft':
    X = zeros((T,d))
    title = 'FFT y=%s A=%s' % (basename(args.file), basename(args.base))
    for t,x in enumerate(fft_infer(Y)):
        print '%d/%d' % (t+1,T)
        X[t] = x
    viz(X.T, notes, sample_rate, window_size, save=1, delay=0)
    exit()


if args.by=='nmf':

    title = 'NMF euclid y=%s A=%s (online)' % (basename(args.file), basename(args.base))
    X = zeros((T,d))
    for t in range(T):
        X[t] = nmf(A,Y[t], iters=10)
        viz(X.T, notes, save=0, delay=0, title=title)
    viz(X.T, notes, save=1, title=title)

    # iters = 50
    # title = 'NMF euclid y=%s A=%s iters=%d (global)' % (basename(args.file), basename(args.base), iters)
    # X = nmf(A,Y, iters=iters)
    # viz(X, notes, save=1, title=title)

    exit()


def X0(): return [0]*d
X = zeros((T,d), dtype=bool)
bef()
for t,x in enumerate(particle_filter(Y, X0, sample, weigh, L=args.L, K=args.K, gamma=args.gamma)):
    X[t] = x
    if (t+1) % (2*window_rate) < 1: # about every second (nb. window_rate is not an int)
        clf()
        viz(X.T, notes, sample_rate, window_size, save=0, title='', delay=0)
print 'runtime = %ds' % aft()

clf()
title = 'polytrans y=%s A=%s L=%d K=%d gamma=%s (p00, p11)=(%s, %s)' % (
    basename(args.file), basename(args.base), args.L, args.K, args.gamma, p00, p11)
viz(X.T, notes, sample_rate, window_size, save=1, title=title)


""" TODO



"""
