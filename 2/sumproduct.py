#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf
import networkx as nx
import itertools
from copy import deepcopy
from collections import namedtuple

from factorgraph import *
from sam.sam import *

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

    normalize_messenger(Mu)
    normalize_messenger(Nu)

    return Mu, Nu


def normalize_messenger(X):
    for x in X: X[x] = pd(X[x])



def fac2var(_Mu,Nu, G, f,v):
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
    #print "fac '%s' \t=>\t var '%s'" % (f,v)
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

        _Mu[f,v][val] = msg
    
    """

    # sum (fac * prod nus)
    fac = G.node[f]['pmf']
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


def var2fac(Mu,_Nu, G, v,f):
    #print
    #print "var '%s' \t=>\t fac '%s'" % (v,f)
    assert G.type(v)=='var' and G.type(f)=='fac'

    """
    for val in G.vals(v):
        msg = product([ Mu[_f, v][val]  for _f in G.N(v)  if _f != f ])
        Nu[v,f][val] = msg
    """
    msg = [ product([ Mu[_f, v][val]  for _f in G.N(v)  if _f != f ])
            for val in G.vals(v) ]
    
    _Nu[v,f] = msg

    #print 
    #print 'Nu =', Nu


def msg(M,N,G, x,y):
    if G.type(x)=='var':
        var2fac(M,N, G, x,y)
    else:
        fac2var(M,N, G, x,y)



def marginal(Mu, G, v):
    # forall f in G.N(v),  p = Nu[v,f] * Mu[f,v]
    return pd([ product([ Mu[f,v][val] for f in G.N(v) ])  for val in G.vals(v) ])

def marginals(Mu,G, vars):
    return { v : marginal(Mu,G, v) for v in vars }



def marginalize_sumprod(G, 
                        M=1, N=500, P=2, eps=1e-6, vars=None,
                        aux_y=None, aux_f=None, verbose=True):
    """
    : any factor graph => marginals
    : iterative algorithm
     exact and fast on factor trees
     approximate and slow on cyclic factor graphs

    factor graph
    randvars : discrete
    message update schedule : parallel

    [message passing protocol]
    distribute message to some neighbor node
    <->
    have collected message from every other neighbor node

    [message passing protocol]
    : parallel
    each iteration, collect from i-1 => distribute to i

    [iteration]
    set next factor-to-variable msgs  from curr variable-to-factor msgs
    =>
    update variable-to-factor msgs  given factor-to-variable msgs
    =>
    normalize msgs -> avoid underflow numeric

    factor graph ~ undirected graph
    factor graph : tree  <->  factor graph as undirected graph : tree
    -> sumprod on factgraphs iterates approximately (not recurs exactly)
    

    xs = x        ->  marginalize for this var    => [m]
    xs = [x,y..]  ->  marginalize for these vars  => [m..]
    xs = None     ->  marginalize for each var    => [m..]


    'aux_f' is an auxiliary function that can compute whatever it wants every iteration
    'aux_y' is the auxiliary data structure aux_f saves what it computes into
    (later) 'aux_x' are the iteration's data that aux_f computes

    """
    if not vars: vars = G.vars()

    kwargs = [('N',N),
              ('eps',eps),
              ('vars',vars)]
    alert('sumprod')
    for k,v in kwargs: var(k,v, new=False)

    if aux_y and aux_f:
        Aux = namedtuple('Aux', ['G', 'Mu','Nu'])

    # "_X" set/write to next/new
    # "X" get/read from curr/old
    _Mu, _Nu = make_messengers(G)
    Mu, Nu = deepcopy(_Mu), deepcopy(_Nu)

    i = 0
    _diff, diff = +inf, 0
    stuck = 0
    while True:
        if not i < N:
            alert('[sumprod: iterated too many times (N=%d) with (diff=%.9f)]' % (N,diff))
            break

        if M < i:
            if abs(_diff - diff) < eps and stuck > 1: #HACK diff hits zero every other dunno why
                print '[sumprod: converged (eps=%.0e) in %d iterations]' % (eps, i)
                break

            if stuck > P:
                print '[sumprod: got stuck %d times at diff=%.9f]' % (P, diff)
                break

        i += 1
        if verbose: print; print i

        if aux_y and aux_f:
            aux_x = Aux(G=G, Mu=Mu, Nu=Nu)
            aux_f(aux_x, aux_y)

        # parallel update schedule

        # factor-to-variable
        for f in G.facs():
            for v in G.N(f):
                msg(_Mu,Nu,G, f,v)
        normalize_messenger(_Mu)

        # variable-to-factor
        for v in G.vars():
            for f in G.N(v):
                msg(_Mu,_Nu,G, v,f)
        normalize_messenger(_Nu)
        
        # var('Mu',Mu)
        # var('_Mu',_Mu)
        _diff = max( max( max(abs(_Mu[fv] - Mu[fv])) for fv in Mu ),
                     max( max(abs(_Nu[vf] - Nu[vf])) for vf in Nu ))
        
        if verbose: var('diff', '%.12f' % abs(_diff - diff))

        stuck = 1+stuck if abs(_diff - diff) < eps else 0

        diff   = _diff
        Mu, Nu = deepcopy(_Mu), deepcopy(_Nu)

    if aux_y is not None and aux_f is not None:
        return marginals(Mu,G, vars=vars), aux_y
    else:
        return marginals(Mu,G, vars=vars)



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


def marginalize_bruteforce(G, vars=None):
    if not vars: vars = G.vars()
    
    p,_ = joint(G)
    def but(i): return tuple([j for j in range(len(G.vars())) if j!=i])

    marginals = { v : p.sum(axis=but(i))
                  for i,v in enumerate(G.vars()) 
                  if v in vars }

    return marginals

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# C


def test(G, **kwargs):

    var('G', G.node)

    bf = marginalize_bruteforce(G)
    sp = marginalize_sumprod(G, **kwargs)

    var('bf', bf)
    var('sp', sp)

    compare(sp,bf)


def compare(ps,qs, fail=True):
    print

    if fail:
        for var,p,q in zip( ps, ps.values(), qs.values() ):
            assert all([near(pi,qi) for pi,qi in zip(p,q)])
        return True

    else:
        return all([ all([near(pi,qi) for pi,qi in zip(p,q)])  for var,p,q in zip(ps, ps.values(), qs.values()) ])


if __name__=='__main__':

    div('testing bruteforce marginalize')

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

    div('testing sumprod marginalize')



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
    var('sp', sp)

    bf = marginalize_bruteforce(G)    
    var('bf', bf)

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
    var('sp', sp)

    bf = marginalize_bruteforce(G)    
    var('bf', bf)

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
    var('sp', sp)

    bf = marginalize_bruteforce(G)    
    var('bf' , bf)

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
    var('sp', sp)

    bf = marginalize_bruteforce(G)    
    var('bf', bf)

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
    var('sp', sp)

    bf = marginalize_bruteforce(G)    
    var('bf', bf)

    compare(sp,bf)



    div('testing general iterative sumprod')

    print;print;print 'testing...'
    G = factor_1f3v()
    test(G)
    
    print;print; alert('testing...')
    G = factor_3f1v()
    test(G)

    print;print; alert('testing list...')
    G = factor_list()
    test(G)

    print;print; alert('testing tree...')
    G = factor_tree()
    test(G)
    
    print;print; alert('testing square graph with a cycle...', t=0)
    G = factor_square()
    test(G)
    
    print;print; alert('testing binary tree, whether it converges in |depth| iters...', t=0)
    G = factor_btree()
    test(G)

    print
    print
    print 'all tests passed!'
    

