#!/usr/bin/python
from __future__ import division
from sam.sam import *

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf
import networkx as nx
import itertools

from factor_graph import *

"""
[1]

(A)
Implement the sum-product algorithm. Your code should support an arbitrary factor graph
linking a collection of discrete random variables. Use a parallel message update schedule,
in which all factor-to-variable messages are updated given the current variable-to-factor
messages, and then all variable-to-factor messages given the current factor-to-variable
messages. Initialize by setting the variable-to-factor messages to equal 1 for all states.
Be careful to normalize messages to avoid numerical under flow.

(B) Write code which explicitly computes a table containing the probabilities of all joint config-
urations of the variables in a factor graph. Also write code which sums these probabilities
to compute the marginal distribution of each variable. Such "brute force" inference code
is of course inefficient, and will only be computationally tractable for small models.

(C) Create a small, tree-structured factor graph linking four variable nodes. Use this model to
verify that the algorithms implemented in parts (a) and (b) are consistent with each other,
and compute correct marginal distributions. Design your factor graph to validate many
aspects of your code, including variables with different numbers of states, and factors of
varying orders. Clearly describe the experimental evidence you use to verify that your
implementations are correct


"""

cartesian = itertools.product

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# A

def make_messengers(G):

    #: factor , variable => [num]
    # fac to var
    Mu = {}
    for f in G.facs():
        for v in G.N(f):
            Mu[f,v] = array([1 for _ in G.vals(v)])

    #: variable , factor => [num]
    # var to fac
    Nu = {}
    for f in G.facs():
        for v in G.N(f):
            Nu[v,f] = array([1 for _ in G.vals(v)])

    normalize_messenger(G, Mu)
    normalize_messenger(G, Nu)

    return Mu, Nu


def normalize_messenger(G, M):
    for x in M: M[x] = pd(M[x])



def fac2var(Mu,Nu, G, f,v):
    """
    eg
    VAR has { val ... }
    N(F) - X = { Y Z }
    forall x in X,  mu[F,X] x = sum[y,z] f x y z * nu[Y, F] y * nu[Z, F] z

    prob. must preserve order of vars (and their vals) for factor and for consistency
    soln. { var => [vals] }

    ndarray.flatten()
    ==
    ndarray.resize(ndarray.size)
    ==
    ndarray.shape = (ndarray.size,)
    
    """
    #print
    print "fac '%s' \t=>\t var '%s'" % (f,v)
    assert G.type(f)=='fac' and G.type(v)=='var'

    vars = G.N(f) # for order
    ii = { x:i for i,x in enumerate(vars) } # inverted index

    for val in G.vals(v): # forall val in var
        
        # "pin down msg var to one val"
        # eg
        # var = 'b'
        # val = 2
        # vars = ['a','b','c']
        # space = {0..1} x {2} x {0..3}
        space = cartesian( *[(G.vals(_v) if _v != v else [val])  for _v in vars] )

        # get _val of _var
        #  _vals[ii[_v]] = _v:str => ii:str=>inx => _vals:inx=>val => Nu[_,_]:val=>num
        #  discrete randvar -> values are indices
        # sum of prod
        msg = sum(  G(f, *_vals)  *  product([ Nu[_v, f][_vals[ii[_v]]]  for _v in G.N(f)  if _v != v  ])
                    for _vals in space )

        Mu[f,v][val] = msg
    
    """

    # sum (fac * prod nus)
    fac = G.node[f]['p']
    nus = [ (i, _v, Nu[_v, f])  for i,_v in enumerate(G.N(f))  if _v != v  ]

    msg = fac
    for i,_v,nu in nus:
        # sans broadcast
        shape = [1 for _ in msg.shape]
        shape[i] = G.node[_v]['d']
        nu = resize(nu, tuple(shape))
        nu = resize(nu, msg.shape)
        msg = msg * nu
        # [diff] msg = msg * resize(nu, msg.shape)
        # [diff] msg = resize(nu, msg.shape) * msg

    others = tuple([ i  for i,_v in enumerate(G.N(f))  if _v != v ])        
    msg = sum(msg, axis=others) # marginalize every other var
    Mu[f,v] = msg

    """

    #print 
    #print 'Mu =', Mu


def var2fac(Mu,Nu, G, v,f):
    #print
    print "var '%s' \t=>\t fac '%s'" % (v,f)
    assert G.type(v)=='var' and G.type(f)=='fac'

    """
    for val in G.vals(v):
        msg = product([ Mu[_f, v][val]  for _f in G.N(v)  if _f != f ])
        Nu[v,f][val] = msg
    """
    msg = [ product([ Mu[_f, v][val]  for _f in G.N(v)  if _f != f ])
            for val in G.vals(v) ]
    
    Nu[v,f] = msg

    #print 
    #print 'Nu =', Nu


def msg(Mu,Nu,G, x,y):
    if G.type(x)=='var':
        var2fac(Mu,Nu,G, x,y)
    else:
        fac2var(Mu,Nu,G, x,y)



def marginal(Mu, G, v):
    # forall f in G.N(v),  p = Nu[v,f] * Mu[f,v]
    return pd([ product([ Mu[f,v][val] for f in G.N(v) ])  for val in G.vals(v) ])


def sumprod(G, Mu=None, Nu=None):
    """
    : factor graph => marginals

    G = factor graph

    factor graph
    randvars : discrete
    message update schedule : parallel

    (message passing protocol)
    distribute message to some neighbor node
    <->
    have collected message from every other neighbor node

    set next factor-to-variable msgs  from curr variable-to-factor msgs
    =>
    update variable-to-factor msgs  given factor-to-variable msgs
    =>
    normalize msgs -> avoid underflow numeric

    factor graph ~ undirected graph
    factor graph : tree  <->  factor graph as undirected graph : tree
    -> sumprod on factgraphs iterates approximately (not recurs exactly)
    
    
    """
    if not Mu or not Nu:
        Mu, Nu = make_messengers(G)

        

    return Mu, Nu
    

def marginalize_sumprod(G, xs=None, tree=True):
    """
    G : any factor graph

    exact and fast on factor trees
    approximate and iterative on cyclic factor graphs

    xs = x        ->  marginalize for this var    => [m]
    xs = [x,y..]  ->  marginalize for these vars  => [m..]
    xs = None     ->  marginalize for each var    => [m..]

    """

    if tree:
        return sumprod(G)

    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# B

def joint(G, xs=None):
    """

    CASE
    same vars in diff factors
    (in particular, two factors on same vars should just be multiplied and renormalized)
    eg joint p(x,y) q(y,z) => pq(x,y,z) not pq(x,y,y,z)
    
    
    """
    vars = G.vars() #: [var]
    facs = { f : G.N(f) for f in G.facs() } #: fac => vars

    dims = [G.node[x]['d'] for x in vars] #: [nat]
    _joint = ones(dims)

    for vals in itertools.product( *(xrange(d) for d in dims) ): # cartesian product
        _vars = dict(zip(vars,vals)) #: var => val
        vals = tuple(vals) # to index
        #print
        #print _vars
        for fac in facs:
            _vals = [_vars[v] for v in facs[fac]] # keep only fac's vars' vals
            #print '%s%s' % (fac, tuple(_vals))
            _joint[vals] *= G(fac, *_vals)

    Z = sum(_joint)

    return pd(_joint), Z


def marginalize_bruteforce(G, xs=None):
    p,_ = joint(G)
    def but(i): return tuple([j for j in range(len(G.vars())) if j!=i])

    marginals = { v : p.sum(axis=but(i))  for i,v in enumerate(G.vars()) }

    return marginals

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# C


def test(G):

    Mu,Nu = make_messengers(G)
    
    print
    print 'G =', G.node

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    sp = marginalize_sumprod(G)
    print
    print 'sp =', sp

    compare(sp,bf)


def compare(ps,qs):
    for var,p,q in zip( ps, ps.values(), qs.values() ):
        assert all([near(pi,qi) for pi,qi in zip(p,q)])


if __name__=='__main__':

    print
    print
    print '--- testing bruteforce marginalize ---'

    print
    print
    T = factor_tree()
    pT,Z = joint(T)
    assert pT.shape == (1,2,3,4)
    assert near( pT[0,0,0,0] ,
                 T('f[ac]',0,0) * T('f[bc]',0,0) * T('f[cd]',0,0) *
                 T('f[a]',0) * T('f[b]',0) * T('f[d]',0) *
                 (1/Z)
                 )
    assert near( pT[0,1,2,3] ,
                 T('f[ac]',0,2) * T('f[bc]',1,2) * T('f[cd]',2,3) *
                 T('f[a]',0) * T('f[b]',1) * T('f[d]',3) *
                 (1/Z)
                 )

    print
    print
    L = factor_list()
    pL,Z = joint(L)
    assert pL.shape == (1,2,3,4)
    assert near( pL[0,0,0,0] ,
                 L('f[ab]',0,0) * L('f[bc]',0,0) * L('f[cd]',0,0) *
                 (1/Z)
                 )
    assert near( pL[0,1,2,3] ,
                 L('f[ab]',0,1) * L('f[bc]',1,2) * L('f[cd]',2,3) *
                 (1/Z)
                 )
    
    print
    print
    C = factor_clique()
    pC,Z = joint(C)
    assert pC.shape == (1,2,3,4)
    assert near( pC[0,0,0,0] , C('f[abcd]', 0,0,0,0) * (1/Z))
    assert near( pC[0,1,2,3] , C('f[abcd]', 0,1,2,3) * (1/Z))

    print
    print
    print '--- testing sumprod marginalize ---'



    print
    print
    print 'testing 3 facs 1 var...'
    G = factor_3f1v()
    Mu,Nu = make_messengers(G)
    m = implicit(Mu,Nu,G)(msg)

    m('f1','v')
    m('f2','v')
    m('f3','v')

    m('v','f1')
    m('v','f2')
    m('v','f3')

    sp = {v:marginal(Mu,G, v) for v in G.vars()}
    print
    print 'sp =', sp

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    compare(sp,bf)



    print
    print
    print 'testing 1 fac 3 var...'
    G = factor_1f3v()
    Mu,Nu = make_messengers(G)
    m = implicit(Mu,Nu,G)(msg)

    m('a','f')
    m('b','f')
    m('c','f')

    m('f','a')
    m('f','b')
    m('f','c')

    sp = {v:marginal(Mu,G, v) for v in G.vars()}
    print
    print 'sp =', sp

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    compare(sp,bf)

    
    
    print
    print
    print 'testing small list...'
    G = factor_small()
    Mu,Nu = make_messengers(G)
    m = implicit(Mu,Nu,G)(msg)
    
    m('f[a]', 'a')
    m('a','f[ab]')
    m('f[ab]', 'b')
    
    m('b', 'f[ab]')
    m('f[ab]','a')
    m('a', 'f[a]')

    sp = {v:marginal(Mu,G, v) for v in G.vars()}
    print
    print 'sp =', sp

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    compare(sp,bf)
    


    print
    print
    print 'testing list...'
    G = factor_list()
    Mu,Nu = make_messengers(G)
    m = implicit(Mu,Nu,G)(msg)

    m('a','f[ab]')
    m('f[ab]', 'b')
    m('b','f[bc]')
    m('f[bc]', 'c')
    m('c','f[cd]')
    m('f[cd]','d')

    m('d','f[cd]')
    m('f[cd]','c')
    m('c', 'f[bc]')
    m('f[bc]','b')
    m('b', 'f[ab]')
    m('f[ab]','a')

    sp = {v:marginal(Mu,G, v) for v in G.vars()}
    print
    print 'sp =', sp

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    compare(sp,bf)



    print
    print
    print 'testing tree...'
    G = factor_tree()
    Mu,Nu = make_messengers(G)
    m = implicit(Mu,Nu,G)(msg)
    #nx.draw(G); show()

    # start at leaves of tree
    m('f[a]','a')
    m('f[b]','b')
    m('f[d]','d')
    
    m('a','f[ac]')
    m('b','f[bc]')
    m('d','f[cd]')
    
    # goto root of tree
    m('f[ac]','c')
    m('f[bc]','c')
    m('f[cd]','c')
    
    m('c','f[ac]')
    m('c','f[bc]')
    m('c','f[cd]')
    
    m('f[ac]','a')
    m('f[bc]','b')
    m('f[cd]','d')
    
    m('a','f[a]')
    m('b','f[b]')
    m('d','f[d]')
    
    sp = {v:marginal(Mu,G, v) for v in G.vars()}
    print
    print 'sp =', sp

    bf = marginalize_bruteforce(G)    
    print
    print 'bf =', bf

    compare(sp,bf)




    print
    print
    print 'all tests passed!'
    

