#!/usr/bin/python -W ignore::Warning
from __future__ import division
from numpy import *
from matplotlib.pyplot import *
import numpy.random as samples
import scipy.stats as pdfs

import argparse
import time
from itertools import groupby

from util import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Infer

def particle_filter(Y, X0, sample, weigh, L=100):
    """
    a simple bootstrap particle filter approximately solves any state space model
    samples from likelihood and weighs by transition
    Y is very informative
    X is just for smoothness, sparsity, noise
    
    for each moment
    particles <== sample _
    weights <== weigh _ _
    resample from multinomial(particles, weights)

    Y , data : stream

    X0
    can sample
    : => X

    sample
    eg p(y[t] | x[t])
    can sample
    : _ => _

    weigh
    eg p(x[t] | x[t-1])
    can eval
    : _, _ => [0,1]

    L = |particles|

    output X
    T lists of L particles with d dimensions
    X[t,l,d]
    """
    print '--- Particle Filter (L=%d) ---' % L

    Xs = a([X0() for _ in range(L)])
    L,d = Xs.shape
    ymax = 1e6

    for t,y in enumerate(Y):
        print '- %d -' % (t+1)
        ymax = max(ymax, y.max())

        # sample
        particles = sample(y, L, ymin=0, ymax=ymax)

        weights = a([weigh(Xs[l], particles[l]) for l in range(L)])
        weights /= sum(weights)

        # resample
        Xs = a([multinomial(particles, weights) for _ in range(L)])

        # infer
        X = [max(var, key=lambda val: sum(val==Xs[:,_])) for _ in range(d)]
        yield X


def fft_infer(Y):
    for y in Y:
        _freqs = {i*window_rate : y[i] for i in range(len(y))} # y[fft basis] => y[freq basis]
        _notes = {n : max(_freqs[f] for f in fs) # max ampl of freqs near note
                  for n,fs in groupby(sorted(_freqs),key=note)} # group freqs by note
        yield a([_notes.get(notes[j],0) for j in range(d)])

def nmf(A, B, iters=50):
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
        print '%d/%d' % (i+1,iters)
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

#TODO data-driven
p00 = 0.05 # P(silence sticks)
p11 = 0.999 # P(sound sticks)
transition = a([[p00, 1-p00],
                [1-p11, p11]])
def weigh(prevs, currs):
    #TODO vectorize
    return product([transition[prev,curr] for prev,curr in zip(prevs,currs)])

def sample(y, L, ymin=1e4, ymax=+inf):
    """
    : audio => note

    min/max energies
    changes greatly given different audio, format, recording, processing, etc
    smoothly between . absolute metric for low energies . relative metric for high energies

    y[i]
    fft is wrt cycles / samples/window  but  freqs are wrt cycles / second
    (cycles / samples/window) * (1 / windows/second) * (samples / seconds) = cycles / second = Hz

    highest-amplitude freqencies
    uniq([note(i * (sample_rate / window_size))  for i in y.argsort().tolist()[::-1] if y[i]>0])

    runtimes (d=44)
    ./polytrans.py y/chord.wav base/octave/
    T(L=1)      = 0.30s
    T(L=10)     = 0.30s
    T(L=100)    = 0.30s
    T(L=1000)   = 0.35s
    T(L=10000)  = 1.75s

    """
    p = len(y)

    _freqs = {i*window_rate : y[i] for i in range(p) if y[i]>ymin} # y[fft basis] => y[freq basis]
    _notes = {n : max(_freqs[f] for f in fs) / ymax # max ampl of freqs near note
              for n,fs in groupby(sorted(_freqs),key=note)} # group freqs by note

    return a([[ber(_notes.get(notes[j],0)) for j in range(d)] for _ in range(L)])



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Main

Y, sample_rate = fft_wav(args.file, window_size=window_size)
T, _ = Y.shape # time in windows
window_rate = sample_rate / window_size # samples/second / samples/window = windows/second


if args.by=='fft':
    X = zeros((T,d))
    title = 'FFT y=%s A=%s' % (basname(args.file), basename(args.base))
    for t,x in enumerate(fft_infer(Y)):
        print '%d/%d' % (t+1,T)
        X[t] = x
    viz(X.T, notes, sample_rate, window_size, save=1, delay=0)
    exit()


if args.by=='nmf':
    X = zeros((T,d))
    iters = 50
    title = 'NMF euclid y=%s A=%s iters=%d' % (basname(args.file), basename(args.base), iters)
    for t in range(T):
        X[t] = nmf(A,Y[t], iters=10)
        viz(X.T, notes, save=0, delay=0, title=title)
    viz(X.T, notes, save=1, title=title)
    exit()


bef()
def X0(): return [0]*d
X = zeros((T,d), dtype=bool)
for t,x in enumerate(particle_filter(Y[:40], X0, sample, weigh, L=args.L)):
    X[t] = x
    if t % (2*window_rate) < 1: # about every second (nb. window_rate is not an int)
        clf()
        viz(X.T, notes, sample_rate, window_size, save=0, title='', delay=0)
print 'runtime = %ds' % aft()

clf()
title = 'polytrans y=%s A=%s L=%d (p00, p11)=(%s, %s)' % (basename(args.file), basename(args.base), args.L, p00, p11)
viz(X.T, notes, sample_rate, window_size, save=1, title=title)

