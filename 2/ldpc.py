#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf
from scipy.io import loadmat
import networkx as nx
from scipy.ndimage import imread

from sam.sam import *
import sam.sam as sam
from factorgraph import *
from sumproduct import *

image = 1

runA = 0
runB = 0
runC = 0
runD = 0
runE = 1
runF = 1

"""


"""


""" 1a
: ldpc code => FGM

factor : predicate

test  P(invalid)==0

"""

def MLE(marginals):
    """ : {X : pmf on X} => {X => val of X} """
    return { X:argmax(pX)  for X,pX in marginals.items() }

def isBitvector(B):
    return type(B)==ndarray and all((B==0) + (B==1))

def decode(Ps):
    """ decode message from marginals
    : { var => [(val,prob)] } => [val]
    """
    return array([ val for (var,val) in sorted( MLE(Ps).items(), key=lambda(var,val):var ) ])

def ldpc_to_fgm(H, debug=False):
    """ : ldpc code matrix => factor graphical model
    H[i,j] == 1  :=  bit j depends on check i

    
    """
    
    n_facs, n_vars = H.shape

    G = FactorGraph()
    G.graph['ldpc'] = True
    def name(j): return 'b%d' % j

    for j in range(n_vars): G.add_var(name=name(j), d=2)
    #  bits are binary randvars named 'b0'..
    for v in G.vars(): G.add_fac(p=[0.5, 0.5], vars=[v])
    #  before set_bits, assume nothing (uniform prior)
    
    for i,row in enumerate(H):
        js = [j for j,active in enumerate(row) if active]
        vars = [name(j) for j in js]

        fac = lambda *vals: sum(vals) %2 ==0
        fac.__name__ = 'f%s' % (str(js).replace(' ',''))

        G.add_fac(p=fac, vars=vars, name=fac.__name__)

    if debug:
        alert(':ldpc => :fgm')    
        # var('facs', G.facs())
        # var('vars', G.vars())

    return G


def prior0(eps): return pd([1-eps, eps])
def prior1(eps): return pd([eps, 1-eps])

def set_bits(G,B,eps):
    B = array(B)

    assert 'ldpc' in G.graph, 'set_bits(G,_,_): G must be a LDPC graph'
    assert isBitvector(B), 'set_bits(_,B,_): B must be bitvector'
    assert 0<= eps <=1, 'set_bits(_,_,eps): eps must be in [0,1]'

    priors = { G.node[fac]['vars'][0]: fac
               for fac in G.facs() if len(G.node[fac]['vars'])==1 }

    for var,bit in zip(G.vars(), B):
        # change val
        G.node[var]['val'] = bit

        # change prior
        fac = priors[var]
        G.node[fac]['pmf'] = prior1(eps) if bit==1 else prior0(eps)


""" 1b
msg = zeros(N)

make binary symmetric channel
"""

def coin(p=0.5): return random.random() < p

def flip(bit): return (bit+1) % 2

def channel(bits, eps=0.01): # eps=0.01 -> P(some bit flips) = 1%
    return array([ flip(bit) if coin(eps) else bit  for bit in bits ])

""" 1c
[maybe change the prior, if input isnt always zero vector]

"""

def hamming_distance(x,y):
    assert isBitvector(x) and isBitvector(y) and x.shape==y.shape
    return sum(abs(x-y))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Test

def valid(G):
    # only higher factors
    checks = [f for f in G.facs()  if len(G.node[f]['vars']) > 1]

    # bits are in the vals
    return [G(f, *[G.node[v]['val'] for v in G.node[f]['vars']])
            for f in checks]

def bitstring(B): return ''.join([str(int(b)) for b in B])

def test(G, eps=None, N=None):
    assert N is not None and eps is not None
    
    var('eps', eps)
    var('sparsity H', sum(H) / (n * 2*n))

    input = zeros(2*n) # 0..0 valid under any parity check
    text('channel outputs this.')
    B = channel(input, eps=eps)
    set_bits(G, B, eps)
    var('B', B)
    
    text('channel got what input?')
    Ps = marginalize_sumprod(G, N=N, verbose=1)
    print(Ps)

    posterior_is_one = [probs[1] for (bit,probs) in 
                        sorted(list(Ps.items()), key=fst)]

    # observed channel output
    X = B
    # decoded channel output as most probable input
    fX = [int(p>0.5) for p in posterior_is_one]
    # true channel input
    Y = input

    var('valid output?', all(valid(G)))
    var('decode(output) == true input?', all(fX==Y))
    #var('output', bitstring(fX))
    #var('input', bitstring(Y))

    print()
    text('decode(output) versus true input')
    color('green', 'o'); print(':= same')
    color('red', 'x');   print(':= diff')
    for fx,y in zip(fX, Y):
        if fx==y:
            color('green', 'o')
        else:
            color('red', 'x')

    return X,fX,Y, Ps,posterior_is_one
    

def tests(K=None, N=None, eps=None, name=None):
    assert all([ arg is not None for arg in [K,N,eps,name] ])

    Y = zeros(2*n)

    def aux_f(aux_x, aux_y):
        G,Mu = aux_x.G, aux_x.Mu
        Ps = marginals(Mu,G)
        fX = array(list(MLE(Ps).values()))
        aux_y.append(hamming_distance(Y,fX))
        #  order dont matter, since Y is all zeros
    
    for i in range(K):
        text(str(i+1))
        B = channel(Y, eps=eps)
        set_bits(G, B, eps)
            
        # marginals and hamming distances
        Ps, Ds = marginalize_sumprod(G, N=N,
                                         aux_f=aux_f, aux_y=[], verbose=0)

        if image:
            figure()
            x,xx = 1,N
            y,yy = 0 -1, n +1
            axis((x,xx,y,yy))
            scatter([j+1 for (j,_) in enumerate(Ds)], Ds, s=50)
            savefig('img/%s/%s.png' % (name, sam.pad(i+1, '0', 2)))


def bwshow(bool_matrix):
    imshow(bool_matrix, cmap='Greys', interpolation='nearest')

def img2msg(img):
    msg = img.reshape(1600)
    msg = odd(mul(G, msg))
    return msg

def msg2img(msg):
    return msg[:1600].reshape((40,40))

# 1e 1f    
def decode_image(dir, img, eps):
    ion()

    data = loadmat('data/ldpc36-1600.mat')
    G,H,_img = data['G'],data['H'],data['logo']
    n = 1600
    assert all(G[:n,:] == identity(n))
    # G encodes any image wrt H
    # G is half identity, half matrix that magically
    #  multiplies any image into itself concatenated 
    #  onto some bits such that together they
    #  satisfy all the parity checks of H

    img = imread('data/%s' % img)
    img = img < (img.max() - img.min())/2 # black and white
    height, width = img.shape
    assert height==40 and width==40
    assert all(img == img.reshape(1600).reshape((40,40)))
    #imshow(img, cmap='Greys', interpolation='nearest') ;show()

    ion()
    img = _img
    bwshow(img);draw()

    ## image => encode by G => noise by channel => decode by sumprod => cmp

    _msg = img.reshape(1600) # before
    _msg = odd(G.dot(_msg))
    #msg = channel(_msg, eps) # after
    msg = _msg # DOESNT HELP
    #bwshow(msg2img(msg));draw();
    bwshow(msg.reshape((80,40)));draw()
    var('hamming x y / len(x or y)', hamming_distance(_msg, msg) / len(msg))

    # circ = zeros((7,7),dtype=bool)
    # circ[1,3]    =1
    # circ[2,[2,4]]=1
    # circ[3,[1,5]]=1
    # circ[4,[2,4]]=1
    # circ[5,3]    =1

    # msg = circ.reshape(49)
    # print info(msg)
    # print odd.__module__
    # print mul.__module__
    # print 'segfault?'
    # msg = odd(mul(G, msg))
    # msg = channel(msg, eps)
    # bwshow(msg.reshape((7,7)));show()


    #time.sleep(600);exit()

    FG = ldpc_to_fgm(H)
    set_bits(FG, _msg, eps)

    N = 50
    Ns = [0, 1, 2, 3, 5, 10, 20, 30]
    def aux_f(aux_x, aux_y):
        # # init to prior
        # # DOESNT HELP
        # if aux_x.i == 1:
        #     print aux_x.Mu[('f0','b0')]
        #     G = aux_x.G
        #     for j in range(3200):
        #         b = 'b%d'%j
        #         f = 'f%d'%j
        #         print G.node[f]
        #         if G.node[f]['pmf'].size == 2:
        #             aux_x.Mu[(f,b)] = G.node[f]['pmf']

        # get marginals => decode => msg to img => save/show
        if aux_x.i in Ns:
            Ps = marginals(aux_x.Mu, aux_x.G)
            msg = decode(Ps)
            img = msg.reshape((80,40))
            aux_y.append( img )
            figure(); bwshow(img) ;draw()
            savefig('img/%s/%s.png' % (dir, sam.pad(N,'0',2)))

    kwargs = dict(N=N, aux_f=aux_f, aux_y=[], verbose=1)
    Ps, aux_y = marginalize_sumprod(FG, **kwargs)

    # time.sleep(60*60)
    return FG, Ps, aux_y

if __name__=='__main__':
    
    div('1A')
    if runA:
        eps = 0
    
        H = zeros((3,6), dtype=uint8)
        H[0,:] = [1,0,0,0,0,1]
        H[1,:] = [0,1,0,1,0,1]
        H[2,:] = [1,0,1,0,1,0]
        var('H',H)
    
        B = [1, 0, 1, 1, 0, 1]
        var('B',B)
    
        G = ldpc_to_fgm(H)
        set_bits(G,B,eps)
    
        var('G',G.node)
    
        checks = [f for f in G.facs()  if len(G.node[f]['vars']) > 1]
        var('checks', checks)
    
        text("it's a valid codeword")
        ok = valid(G)
        print(ok)
        assert all(ok)
    
        text("if you flip any one bit, some parity check fails, and the message becomes invalid")
        for v in G.vars():
            G.node[v]['val'] = flip(G.node[v]['val'])
    
            ok = valid(G)
            print(ok)
            assert not all(ok)
            
            
            G.node[v]['val'] = flip(G.node[v]['val'])




    H = loadmat('data/ldpc36-128.mat')['H']
    n,_ = H.shape
    G = ldpc_to_fgm(H)
    #  its ok to reuse G, so long as you use set_bits


    div('1B')
    if runB:
        N = 50

        X,fX,Y,ms,p1 = test(G, eps=0.05, N=N)

        if image:
            figure()
            x,xx = 1,n
            y,yy = 0,1
            axis((x,xx,y,yy))
            scatter([j+1 for j in range(len(p1))], p1, s=50)
            savefig('img/%s/%s.png' % ('1b', 'plot'))


    div('1C')
    if runC:
        tests(K=10, N=50, name='1c', eps=0.05)

    div('1D')
    if runD:
        tests(K=10, N=50, name='1d', eps=0.09)
        

    div('1E')
    if runE:
        decode_image('1e', 'calvin.png', 0.08)

    div('1F')
    if runF:
        # more noise
        decode_image('1f', 'calvin.png', 0.16)


    div('test')
    if test:
        circ = zeros((7,7),dtype=bool)
        circ[1,3]    =1
        circ[2,[2,4]]=1
        circ[3,[1,5]]=1
        circ[4,[2,4]]=1
        circ[5,3]    =1
        bwshow(circ)

    data = loadmat('data/ldpc36-1600.mat')
    G,H,_ = data['G'],data['H'],data['logo']

    div('all tests passed!')


#F, Ps, aux_y = decode_image('1e', 'calvin.png', 0.08)
