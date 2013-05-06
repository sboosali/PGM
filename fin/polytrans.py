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

    X = a(X0)
    d, = X.shape
    ymax = 1e6

    for t,y in enumerate(Y):
        print '- %d -' % (t+1)

        ymax = max(ymax, y.max())

        # sample
        particles = sample(y, L, ymin=0, ymax=ymax)

        # weigh
        weights = a([weigh(X, particle) for particle in particles])
        weights /= sum(weights)

        # resample
        Xs = a([multinomial(particles, weights) for _ in range(L)])

        # infer
        X = [max(var, key=lambda val: sum(val==Xs[:,_])) for _ in range(d)]
        yield X


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Input

cmd=argparse.ArgumentParser(description='Polyphonic Transcription by Particle Filter with Likelihood Samples')
cmd.add_argument('file', help='a .wav audio file, the input to transcribe')
cmd.add_argument('data', help='a dir of .wav audio files, defines what notes are')
args=cmd.parse_args()

file = args.file
window_size = 2**12 # 44100 samples/second / 2^12 samples/window = 10 windows/second
Y, sample_rate = fft_wav(file, window_size=window_size)
T, _ = Y.shape # time in windows
window_rate = sample_rate / window_size # samples/second / samples/window = windows/second

A,freqs,notes = basis(args.data, truncate=44100*5, window_size=window_size)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model

# the graphical model is a factoiral HMM
var = [0,1]
d,_ = A.shape

L = 100
X0 = [0]*d

#TODO data-driven
p00 = 0.05 # P(silence sticks)
p11 = 0.99 # P(sound sticks)
transition = a([[p00, 1-p00],
                [1-p11, p11]])
def weigh(prevs, currs):
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
    uniq([note(i * (sample_rate / window_size))  for i in y.argsort().tolist()[::-1] if y[i]>min])
    
    """
    p = len(y)

    _freqs = {i*window_rate : y[i] for i in range(p)} # y[fft basis] => y[freq basis]
    _notes = {n : max(_freqs[f] for f in fs) / ymax # max ampl of freqs near note
              for n,fs in groupby(sorted(_freqs),key=note)} # group freqs by note

    return a([[ber(_notes.get(notes[j],0)) for j in range(d)] for _ in range(L)])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Test

print 'T =', T
before = time.clock()

X = zeros((d,T), dtype=bool)
for t,x in enumerate(particle_filter(Y, X0, sample, weigh, L=1000)):
    X[:,t] = x
    if t % (2*window_rate) < 1: # about every second
        clf()
        d2(X, freqs, notes, sample_rate, window_size, save=0, title='', delay=0)

after = time.clock()
print 'runtime = %ds' % int(after-before)

clf()
d2(X, freqs, notes, sample_rate, window_size, save=1, title='')
