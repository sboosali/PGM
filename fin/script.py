#!/usr/bin/python -W ignore::Warning
from __future__ import division
from numpy import *
from matplotlib.pyplot import *

from util import *

# # # # # # # # # # # # # # # # # # # # # # # # 
# truth

A,freqs,notes = basis('data/octave/', truncate=sample_rate*5, window_size=window_size)
Y,sample_rate = fft_wav('y/chord.wav', window_size=2**12);y=Y[0]
T,p = Y.shape
d,p = A.shape

X = zeros((T,d))
X[  0: 40+3,[0,2,4]]=1
X[ 40+3:   80+6,[0,2,5]]=1
X[ 80+6:  120+9,[0,3,5]]=1
X[120+9: 160+12,[0,2,4]]=1
X[160+12:200+15,[2,5,7]]=1
X[200+15:240+18,[0,3,5]]=1
X[260-2:280,[1,4]]=1
X[280:300,[1,3]]=1
X[300:340+4,[0,2,4]]=1
viz(X.T, notes, save=0, title='truth', delay=0)


# # # # # # # # # # # # # # # # # # # # # # # # 
# search

window_size = 2**12
window_rate = sample_rate / window_size
Y,sample_rate = fft_wav('y/chord.wav', window_size=window_size)
A,freqs,notes = basis('data/octave/', truncate=sample_rate*5, window_size=window_size)

T,p = Y.shape
d,p = A.shape
y=Y[0]
c=A[0]
e=A[2]
g=A[4]
semilogx(y)

fundamental_notes = uniq([note(i * (sample_rate / window_size))  for i in y.argsort().tolist()[::-1]])
model_notes = filter(lambda x:x in notes, fundamental_notes)
nt = A[notes[model_notes[0]]]
fq = round(freq(model_notes[0]) / window_rate)

yfq = y[fq]
y[fq] = 0
sub = zeros(y.shape)
sub[fq] = yfq
for k in range(1,10): y[fq+k] = 0; y[fq-k] = 0 # makes it obvious
semilogx(y)
semilogx(sub)
y = Y[0]

def plots(*xs):
    xmin, xmax = 0, max(x.size for x in xs)
    ymin, ymax = min(x.min() for x in xs), max(x.max() for x in xs)

    for x in xs:
        clf()
        axis((xmin,xmax , ymin,ymax))
        semilogx(x)
        draw()
        time.sleep(1)

    clf()
    axis((xmin,xmax , ymin,ymax))
    for x in xs:
        semilogx(x)

def scale(Ax,y):
    return (Ax/Ax.max()) * y.max()
yc = y-scale(c,y)
yce = yc-scale(e,yc)
yceg = yce-scale(g,yce)

# find the note whose argmax matches the audio's argmax
ymax = y.max()
y_argmax = y.argmax()
xs = sorted(range(d), key=lambda i: A[i][y_argmax], reverse=True)
Ax = A[xs[0]]
Ax = Ax / max(Ax) * ymax
def positive(x): x[x<0]=0; return x

semilogx(y);draw();time.sleep(1)
semilogx(Ax);draw();time.sleep(1)
semilogx(positive(y - Ax));draw();time.sleep(1)

