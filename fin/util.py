#!/usr/bin/python -W ignore::Warning
from __future__ import division
from numpy import *
from matplotlib.pyplot import *
from matplotlib import cm
import numpy.random as samples
import scipy.stats as pdfs

from glob import glob
import re
from scipy.io import wavfile

from sam.sam import *


def cat(*As): return concatenate(As)

def mul(*As):
    """ left-associate matrix multiplication """
    prod = 1
    for i,A in enumerate(As): prod = dot(prod,A)
    return prod

def coin(p=0.5): return random.random() < p

def multinomial(xs, ps):
    return xs[ argmax(samples.multinomial(1, ps)) ]

def nans(shape):
    A = empty(shape)
    A.fill(nan)
    return A

def uniq(seq, key=lambda x:x):
    """
    removes duplicates
    key : equivalence relation
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if key(x) not in seen and not seen_add(key(x))]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Piano

K = 0
F = 1
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def make_piano():
    """ : () => [(key, freq)] """
    piano = [('G#0', 25.9565)]
    piano = piano + zip(['A0', 'A#0', 'B0'], [27.5000, 29.1352, 30.8677])
    piano = piano + [ (keys[key] + str(octave), 16.3516 * 2**(octave + key/12))
                      for octave in r(1,7)
                      for key in r(0,11) ]
    piano = piano + [('C8', 4186.01)]
    
    piano = [(k,round(f)) for k,f in piano]
    return piano
piano = make_piano()
assert ('A4', 440) == piano[49]

# runtime should be const
# distance should be log
def note(freq): return reduce( lambda x,y: x if nearer(freq, x[F],y[F]) else y, piano)[K]
assert note(439) == 'A4' == note(441)

def freq(note): return round(dict(piano)[note.upper()])
assert freq('a4') == 440

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Wav

def audio_wav(file, truncate=None):
    """
    reads audio
    
    >>> wavfile.write(file, *wavfile.read(file)) # : identity

    test
    
    """
    #  audio :: |samples| by |channels|
    sample_rate, audio = wavfile.read(file)

    #  keep first channel
    #  keep first 2^31 samples
    if len(audio.shape)==1:
        nSamples, = audio.shape
        nChannels = 1
        audio = audio[:int32(nSamples)]
    else:
        nSamples, nChannels = audio.shape
        audio = audio[:int32(nSamples), 0]

    # consistent times for consistent frequencies
    if truncate:
        audio = audio[:truncate]

    return audio, sample_rate

def my_fft(audio, sample_rate, window_size=2**12):
    """
    does fft on audio

    window_size = 2^n
    fft on power-of-two arrays is much faster

    spectrum : |windows| by window_size
    |frequencies| = window_size
    |windows| / 2 * |frequencies| ~ |seconds| * sample_rate
    
    """

    hanning_window = hanning(window_size)
    n_windows = int32(audio.size/window_size *2) # overlap windows doubles
    spectra = zeros((n_windows, window_size/2)) # symmetric fft halves

    for i in xrange(0,n_windows-1):
        t = int32(i* window_size/2)
        window = audio[t : t+window_size] * hanning_window # elemwise mult
        # half to ignore symmetry => abs to get amplitude
        spectra[i] = abs(fft.fft(window)[:window_size/2])

    return spectra, sample_rate

def fft_wav(file, window_size=2**12, truncate=None):
    """
    reads audio and does fft
    
    >>> file
    >>> Y,sample_rate = fft_wav(file, window_size = 2**12)
    >>> threshold = Y.max() * 0.01
    >>> plots_per_second = 5
    >>> window_size = 2**12
    >>> T,_ = Y.shape
    >>> for t in range(T):
    >>>     if (t % (2*int(sample_rate/window_size) / plots_per_second)):
    >>>         clf()
    >>>         axis((10,window_size/2, int(Y.min()),int(Y.max())))
    >>>         xscale('log')
    >>>         plot(Y[t]*(Y[t]>threshold) + threshold*(Y[t]<threshold))
    >>>         #axhline(threshold)
    >>>         draw(); time.sleep(1/plots_per_second)
    >>> close()

    """
    print 'fft(audio(%s))...' % file
    audio, sample_rate = audio_wav(file, truncate=truncate)
    return my_fft(audio, sample_rate, window_size=window_size)

def file2freq(file):
    """
    eg 'A440.wav' => 440
    """
    file = basename(file)

    frequency = re.match('[A-G][#b]?(?P<freq>[0-9]+)', file)
    if frequency:
        frequency = frequency.groupdict()['freq']
        return int(frequency)

    frequency = re.match('[0-9]+', file)
    if frequency:
        frequency = frequency.group()
        return int(frequency)

    note = re.sub("[^ a-zA-Z #b 0-7]", "", file)
    if note:
        return freq(note)

    raise Exception('bad audio filename "%s"' % file)

def basis(dir, truncate=44100*5, window_size=2**12):
    """
    input dir, a directory of audio files
    
    output A, the basis matrix
    output freqs, a list of frequencies (indexed consistently wrt A)
    output notes, the names of the notes (indexed consistently wrt A)
    """

    data = glob('%s/*.wav' % dir)
    data.sort(key=file2freq)
    n = len(data)
    A = zeros((n, window_size/2)) # : note => spectrum
    freqs = [file2freq(file) for file in data]
    notes = [note(file2freq(file)) for file in data]

    for i,file in enumerate(data):
        spectrum, _ = fft_wav(file, truncate=truncate, window_size=window_size)
        A[i] = sum(spectrum, axis=0) / sum(spectrum)

    return A, freqs, notes

def munge_basis(dir, truncate=44100*5):
    """
    truncate lengths across data
    normalize energies within datum
    """

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot

def pp(x): print x; return x

def viz(X, freqs, notes, sample_rate, window_size, title='', save=True, delay=3600):
    """
    x-axis
    = time in seconds
    window_rate = sample_rate / (sample_size * 2)
    j => j / window_rate
    
    y-axis
    = pitch as note (frequency in Hz)
    i => freqs[i]
    """
    d, n_windows = X.shape
    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(X, cmap=cm.bone_r, origin='lower', aspect='auto', interpolation='nearest')
    axes.set_title(title)

    axes.get_xaxis().set_major_locator(
        LinearLocator(1 + ceil(n_windows/window_rate)))
    axes.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%ds' % round(x/window_rate)))

    axes.get_yaxis().set_major_locator(
                LinearLocator(2*d+1))
    axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%s' % (notes[(y-1)//2] if odd(y) else '')))

    if save:
        ioff()
        show()
    else:
        ion()
        draw()
        time.sleep(delay)
        show()

def bef():
    global before
    before = time.clock()
    return before

def aft():
    global after
    after = time.clock()
    return after-before

