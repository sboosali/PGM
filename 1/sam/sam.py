from __future__ import division

# redefine builtin
from numpy import *
import numpy as np
from numpy.random import multinomial
from matplotlib.pyplot import *
from fractions import Fraction
import random
import time

def about(x,y, eps=0.01): # tolerance within 1%
    if type(x)==np.ndarray and type(y)==np.ndarray:
        return all( abs(x - y) / abs(min(min(x),min(y))) < eps )

    if x==0 or y==0: return False

    return abs(x - y) / abs(min(x,y)) < eps

near = about


def closer(a, x,y):
    """ returns whichever (x or y) is closer to a (x if tie) """
    return y if abs(y-a) < abs(x-a) else x

def nearer(a, x,y):
    """ returns whether x is closer to a than y is (or as close as) (x if tie) """
    return abs(x-a) < abs(y-a) if abs(x-a) != abs(y-a) else True

def half(x): return x[:len(x)//2]

a=array

def smooth(x, eps=1e-9): return x if abs(x) > eps else 0

# inclusive lazy range
def r(x,y): return range(x, y+1)

import sys
def p(x): sys.stdout.write(str(x) + ' ')


def basename(s):
    s = s.split('/')[-1] # rem dirs
    return s[:s.index('.')] # rem ext

#eg [[1],[[2,[3]],[4]],5] => [1,2,3,4,5]
def flatten(l):
    if type(l) != type([]): return [l]
    if l==[]: return []
    return reduce(lambda x,y:x+y, map(flatten,l))

class D(Exception):
    def __init__(self,*args):
        self.env = args
#        self.all = __dict__

def nones(n): return [None for _ in xrange(n)]

def fst(x): return x[0]
def snd(x): return x[1]


# to nearest integer
#:: Real => Int
def round(n): return int(n+0.5)

# generalizes dot from binary to n-ary matrix multiplication
# python reduce = haskell foldl1 = left-associative same-type(seed with first) fold
# *args : tuple
def mul(*args):
    # 1 = lifted multiplicative identity for any size matrix
    return reduce(dot, [1]+list(args))

def odd (n): return n % 2 == 1
def even(n): return n % 2 == 0

def normalize(m):
    low=m.min()
    high=m.max()
    return (m-low)/(high-low)

def t(m): return m.transpose()


def onechannel(audio):
    if len(audio.shape)==1:
        nSamples, = audio.shape
        nChannels = 1
        audio = audio[:int32(nSamples)]
    else:
        nSamples, nChannels = audio.shape
        audio = audio[:int32(nSamples), 0]

    return audio, nSamples, nChannels


def frac(x, d=10):
    return Fraction(x).limit_denominator(max_denominator=d)


def pick(xs, ps):
    return xs[ argmax(multinomial(1, ps)) ]



# make inverted index
def ii(xs): return { x : xs.index(x) for x in xs }

def coin(p=0.5): return random.random() < p


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# bad way of doing joint distribution, but i learned about matrix manipulation

def unzip(xys):
    xs = [x for x,y in xys]
    ys = [y for x,y in xys]
    return xs, ys
    
def argsort(xs, key=lambda x:x):
    """
    indices, elements = argsort(sequence)
    """
    
    js, xs = unzip([(j,x) for (j,x) in sorted(enumerate(xs), key = lambda jx: key(jx[1]))])
    return js, xs

def joint(*args):
    """
    eg
    p(a,b)
    q(c,d,e)
    r(f)

    # associates
    assert joint(p, joint(q,r)) == joint(joint(p,q), r)

    # consistent order
    assert joint(p,q,r)[a,b,c,d,e,f] == p[a,b] * q[c,d,e] * r[f]



    eg...

    p = array([1, 2, 3, 4, 5])
    p = p/sum(p)

    q = array([[ 1,  2],
               [ 3,  4]])
    q = q/sum(q)

    r = array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],

               [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               
               [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]])

    r = r/sum(r)

    rqp = joint(r,q,p)
    assert  r[2,2,2] * q[1,1] * p[0]  ==  rqp[2,2,2, 1,1, 0]
    assert  near(1, sum(rqp))

    """

    #perm, args = argsort(args)

    n = len(args)
    if n==0: return None
    if n==1: return args[0]
    if n==2:
        p,q = args
        return _joint(p,q)

    if n>=3:
        p, qs = args[0], args[1:]
        return _joint(p, joint(*qs))

    

def _joint(p,q):
    """
    eg ...
    
    q =
    [e, f, g]

    p =
    [[a, b],
     [c, d]]

    =>

    qp =
    [[ae, be],
     [ce, de]]
    [[af, bf],
     [cf, df]]
    [[ag, bg],
     [cg, dg]]

    a+b+c+d = 1
    e+f+g = 1
    ->
    a*e+b*e+c*e+d*e + a*f+b*f+c*f+d*f + a*g+b*g+c*g+d*g = 1

    just commute and distribute


    (3,) + (2,2) = (3,2,2)

    if bigger? diff order? both not vectors? 
    

    qp[x,y,z] = q[z] * p[x,y]

    q[2] = g
    p[1,1] = d
    qp[0,1,1] = q[2] * p[1,1] = dg
    
    if not q.ndim < p.ndim:
        t = q
        q = p
        p = t



    p = array([[ 1,  2],
               [ 3,  4]])
    p = p/sum(p)

    q = array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],

               [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               
               [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]])

    q = q/sum(q)

    pq = _joint(p,q)
    assert  p[1,1] * q[2,2,2]  ==  pq[1,1, 2,2,2]
    
    qp = _joint(q,p)
    assert  q[2,2,2] * p[1,1]  ==  qp[2,2,2, 1,1]
    
    """

    p_ = resize(p, p.shape+tuple(1 for _ in range(q.ndim)))
    q_ = resize(q, p.shape+q.shape)
    pq = p_ * q_
    
    return pq






# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def defined(name):
    """
    >>> defined('defined')
    True
    
    >>> defined('undefined')
    False
    """

    return name in locals() or name in globals()


def implicit(*params): # params = decorator params
    def decorator(f): # f = function to decorate
        def decorated(*args): # args = f args
            return f(*(params+args))
        return decorated
    return decorator

def splat(f):
    def decorate(*args):
        return f(args)
    return decorate


def pd(xs):
    xs = array(xs, dtype=np.float)
    xs /= sum(xs)
    assert all([x >= 0 for x in xs])
    assert near(1,  sum(xs))
    return xs


def magic(dims):
    """ not a magic square """
    return array(range(1+0, 1+product(dims))).reshape(dims)




def div(s):
    un = "\033[0;0m" 
    bold = "\033[1m"
    blue = '\033[94m'

    print;print
    print bold+blue + ('--- %s ---' % s) + un

def var(name, val, new=True, tab=False):
    un = "\033[0;0m" 
    bold = "\033[1m"
    green = '\033[92m'

    assert type(name)==str
    
    if new: print
    if tab:
        print bold+green+ name +un + ('\t=\t%s' % str(val))
    else:
        print bold+green+ name +un + (' = %s' % str(val))

def alert(msg, t=0, new=True):
    import time

    un = "\033[0;0m" 
    bold = "\033[1m"
    red = '\033[91m'

    if new: print
    print bold+red+ msg +un
    time.sleep(t)

def timeit(function):
    def decorate(*args):
        before = time.clock()
        y = function(*args)
        after = time.clock()
        t = after - before
        return y,t
    return decorate

