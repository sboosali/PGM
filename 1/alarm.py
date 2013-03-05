#!/usr/bin/python
from __future__ import division
from sam.sam import *

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf

from sumproduct import *
from main import *

"""
run sumprod until
max |message[t] - message[t-1]| < 1e-6
|iterations| > 500


"""

runA=1
runB=1
runC=1

runE=1
runF=1
runG=1
runH=1

VERBOSE = 0


def expectation(p,f, xs):
    return sum( array([p(x) for x in xs]) * array([f(x) for x in xs]) )

def mean(p, xs):
    return dot(p, xs)

def means(G, marginals):
    for v, p in marginals.items():
        var( 'mean %s' % v,  mean(p, [1+x for x in range(G.node[v]['d'])]), new=False, tab=True )
    

def test(H, fail=False, vars=None):
    if not vars: vars = H.vars()

    marginals = marginalize_sumprod(H, vars=vars, verbose=VERBOSE)
    _marginals = marginalize_bruteforce(H, vars=vars)

    var('[sumprod]', marginals)
    var('[bruteforce]', _marginals)

    same = compare(marginals , _marginals, fail=fail)

    if same:
        means(H, marginals)
    else:
        alert('[sumprod] != [bruteforce]');print

        alert('[bruteforce means]')
        means(H, _marginals)

        alert('[sumprod means]')
        means(H, marginals)



""" A
condition on VENTMACH=4-1 and DISCONNECT=2-1
 index 2-1 and 4-1 on all their facs
 renormalize
marginalize all others (i.e. in some fac of some var, but not a var)

sumprod => means of causes
cmp to bruteforce

sumprod: converged (eps=1e-06) in 5 iterations
mean PAP = 2.0079
mean VENTLUNG = 2.10221452209
mean SHUNT = 1.10309499984
mean VENTTUBE = 3.9400000013
mean KINKEDTUBE = 1.96000000177
mean INTUBATION = 1.12999999822
mean PRESS = 2.2487738396
mean PULMEMBOLUS = 1.99

"""
div('A')

if runA:
    causes  = [PULMEMBOLUS, INTUBATION, VENTTUBE, KINKEDTUBE]
    effects = [PAP, SHUNT, PRESS, VENTLUNG]
    vars = causes + effects
    H = G.subgraph(vars, condition={VENTMACH: 4-1, DISCONNECT: 2-1})
   
    test(H, fail=True, vars=causes)




""" B
also condition on SHUNT=2=1 and PRESS=4=3
(python has zero-based indexing, and i index with vals)

sumprod => means of causes
cmp to bruteforce




# same as enumerate-marginals on approx

# mean VENTTUBE = 3.92576957844
# mean KINKEDTUBE = 1.99852287653
# mean INTUBATION = 2.90293068064
# mean PULMEMBOLUS = 1.98107225563

"""
div('B')

if runB:
    causes  = [PULMEMBOLUS, INTUBATION, VENTTUBE, KINKEDTUBE]
    effects = [PAP, VENTLUNG]
    vars = causes + effects
    H = G.subgraph(vars, condition={ VENTMACH: 4-1, DISCONNECT: 2-1, SHUNT: 2-1, PRESS: 4-1 })

    test(H, vars=causes)





""" C
VENTMACH=4-1 DISCONNECT=2-1 PRESS=4-1 MINVOL=2-1

sumprod => means of unobserved
cmp to bruteforce


[bruteforce] means
mean VENTLUNG = 3.42842646489
mean KINKEDTUBE = 1.99787701961
mean INTUBATION = 1.54975162592
mean VENTTUBE = 3.45951755981

sumprod: converged (eps=1e-06) in 105 iterations
[sumprod] means
mean VENTLUNG = 2.66478879532
mean KINKEDTUBE = 1.9931210537
mean INTUBATION = 1.91289992443
mean VENTTUBE = 3.43053893744

"""
div('C')

if runC:
    unobserved  = [INTUBATION, VENTTUBE, KINKEDTUBE, VENTLUNG]
    H = G.subgraph(unobserved, condition={ VENTMACH: 4-1, DISCONNECT: 2-1, PRESS: 4-1, MINVOL: 2-1 })

    test(H, vars=unobserved)

""" D
discuss what caused exact v approx marginals





"""


""" E

probably inexact. the graph is too big

sumprod: converged (eps=1e-06) in 14 iterations
mean DISCONNECT = 1.9
mean PULMEMBOLUS = 1.99001429995
mean INSUFFANESTH = 1.90043976042
mean LVFAILURE = 1.95
mean ANAPHYLAXIS = 1.99106463727

"""
div("E")

if runE:
    H = deepcopy(G)
    H.condition(HYPOVOLEMIA=0, HR=0, INTUBATION=0, KINKEDTUBE=0, VENTALV=0)
    vars = [LVFAILURE, ANAPHYLAXIS, INSUFFANESTH, PULMEMBOLUS, DISCONNECT]
    
    marginals = marginalize_sumprod(H, vars=vars, verbose=VERBOSE)
    means(H, marginals)



""" F


[sumprod: converged (eps=1e-06) in 24 iterations]
mean DISCONNECT = 1.90000000337
mean HYPOVOLEMIA = 1.8
mean LVFAILURE = 1.95
mean KINKEDTUBE = 1.9600000015
mean INTUBATION = 1.12999999528
mean INSUFFANESTH = 1.9
mean ANAPHYLAXIS = 1.99
mean PULMEMBOLUS = 1.99

"""
div("F")

if runF:
    H = deepcopy(G)
    vars = [LVFAILURE, HYPOVOLEMIA, ANAPHYLAXIS, INSUFFANESTH, PULMEMBOLUS, INTUBATION, DISCONNECT, KINKEDTUBE]

    marginals = marginalize_sumprod(H, vars=vars, verbose=VERBOSE)
    means(H, marginals)






""" G

sumprod: iterated too many times (N=500)
mean DISCONNECT = 1.87452095163
mean HYPOVOLEMIA = 1.80217711511
mean LVFAILURE = 1.00441379925
mean KINKEDTUBE = 1.98404363918
mean INTUBATION = 1.65597631809
mean INSUFFANESTH = 1.90559636789
mean ANAPHYLAXIS = 1.99175464545
mean PULMEMBOLUS = 1.99018232094
"""
div("G")

if runG:
    H = deepcopy(G)
    H.condition( HISTORY=0, CVP=0, PCWP=0, BP=0, HRBP=0, HREKG=0, HRSAT=0, EXPCO2=0, MINVOL=0 )
    vars = [LVFAILURE, HYPOVOLEMIA, ANAPHYLAXIS, INSUFFANESTH, PULMEMBOLUS, INTUBATION, DISCONNECT, KINKEDTUBE]

    marginals = marginalize_sumprod(H, vars=vars, verbose=VERBOSE)
    means(H, marginals)






""" H

sumprod: iterated too many times (N=500)
mean DISCONNECT = 1.89900212584
mean HYPOVOLEMIA = 1.80280398627
mean LVFAILURE = 1.00519005955
mean KINKEDTUBE = 1.96375957684
mean INTUBATION = 1.46785352727
mean INSUFFANESTH = 1.89867367916
mean ANAPHYLAXIS = 1.98457028025
mean PULMEMBOLUS = 1.98990789066


"""
div("H")

if runH:
    for v in [HRBP, HREKG, HRSAT]:
        var( 'max %s' % v, G.node[v]['d'] )

    H = deepcopy(G)
    H.condition( HISTORY=0, CVP=0, PCWP=0, BP=0, 
                 HRBP=3-1, HREKG=3-1, HRSAT=3-1,
                 EXPCO2=0, MINVOL=0 )
    vars = [LVFAILURE, HYPOVOLEMIA, ANAPHYLAXIS, INSUFFANESTH, PULMEMBOLUS, INTUBATION, DISCONNECT, KINKEDTUBE]

    marginals = marginalize_sumprod(H, vars=vars, N=2000, verbose=VERBOSE)
    means(H, marginals)

