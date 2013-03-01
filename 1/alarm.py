#!/usr/bin/python
from __future__ import division
from sam.sam import *

from numpy import *
from matplotlib.pyplot import *
import numpy.random as sample
import scipy.stats as pdf

from sum_product import *
from main import *

"""
run sumprod until
max |message[t] - message[t-1]| < 1e-6
|iterations| > 500


"""



effects = [PULMEMBOLUS, INTUBATION, VENTTUBE, KINKEDTUBE]
causes  = [PAP, SHUNT, PRESS, VENTLUNG]
vars = effects + causes
facs = list(set(flatten([G.N(v) for v in vars])))
H = G.subgraph(vars, facs)

def expectation(p,f, xs):
    return sum( array([p(x) for x in xs]) * array([f(x) for x in xs]) )

def mean(p, xs):
    return dot(p, xs)


""" A
condition on VENTMACH=2=1 and DISCONNECT=4=3
 index 2-1 and 4-1 on all their facs
 renormalize
marginalize all others (i.e. in some fac of some var, but not a var)

sumprod => means of causes
cmp to bruteforce
"""

test(H) 

for var,p in ms.items():
    mean(p, [1+x for x in range(G.node[var]['d'])] )


""" B
also condition on SHUNT=2=1 and PRESS=4=3
(python has zero-based indexing, and i index with vals)

sumprod => means of causes
cmp to bruteforce
"""


""" C
VENTMACH=4=3 DISCONNECT=2=1 PRESS=4=3 MINVOL=2=1

sumprod => means of unobserved
cmp to bruteforce
"""

unobserved  = [INTUBATION, VENTTUBE, KINKEDTUBE, VENTLUNG]
facs = list(set(flatten([G.N(v) for v in unobserved])))
H = G.subgraph(unobserved)




if __name__=='__main__':
    pass
