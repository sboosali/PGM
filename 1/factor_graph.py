#!/usr/bin/python
from __future__ import division
from sam.sam import *
from sam import sam

from numpy import *
from matplotlib.pyplot import *
import nltk
import numpy.random as sample
import scipy.stats as pdf
import networkx as nx


"""
init factor graph => add vars => add facs

factor graph : bipartite btwn factors and variables

var
: Maybe Val
has dim
has facts

fact
has potential
has vars
vars = domain potential

repr graph as adjacencies
repr pdfs as numeric potential tables

"""

class FactorGraph(nx.Graph):
    def __init__(self, data=None, **attr):
        """
        factor graph : bipartite btwn factors and variables
        
        node
        | var
        | fac
        
        edge
        : from var to fac
        
        """
        super(FactorGraph, self).__init__(data=data, **attr)

        self.graph['n_vars'] = 0 # number of variables
        self.graph['n_facs'] = 0 # number of factors
        self.graph['vars'] = [] # in deterministic order
        self.graph['facs'] = [] # in deterministic order
        

    def new(self, name):
        n = { 'x': self.graph['n_vars'],
              'f': self.graph['n_facs'],
              }[name]
        
        name = '%s%d' % (name, n)
        
        while name in self:
            n = n+1
            name = '%s%d' % (name, n)
            
        return name
        
    
    def add_var(self, name=None, d=0, val=None):
        """
        variable has...
        v : Maybe Val
        factors fs : [node]
        dimensionality d : nat
    
        eg
        val is unknown
        fs = [X Y]
        d = 3
        """
    
        if val:
            if d:            
                assert d == len(val)
            else:
                d = len(vals)
        
        if name in self:
            raise ValueError('%s in graph' % name)
    
        if name is None:
            name = self.new('x')
    
        self.graph['n_vars'] += 1
        self.add_node( name, d=d, type='var' )
        self.graph['vars'].append( name )
    
    
    def add_fac(self, p, vars, name=None):
        """
        factor has...
        potential p : [real]
        vars = nodes
    
        eg
        p(a,b,c, ..)  :=  potential when A=a, B=b, C=c, ..
        vars = [A B C ..]
    
        
        """
        
        if name is None or name in self:
            name = self.new('f')
    
        p = array(p)
    
        self.graph['n_facs'] += 1
        self.add_node( name, p=p, vars=[], type='fac' )
        self.graph['facs'].append( name )
        
        for x in vars:
            self.add_edge( name, x )
            self.node[name]['vars'].append(x) # for order
    
    
    def N(self, x):
        t = self.node[x]['type']

        if t=='var':
            return [x for x in self.edge[x].keys()]

        if t=='fac':
            return self.node[x]['vars']

        raise ValueError('%s must be var or fac' % x)
    

    def type(self, node):
        return self.node[node]['type']


    def vars(self, but=None):
        #return [x for x in self  if self.node[x]['type']=='var']
        return self.graph['vars']

    def facs(self, but=None):
        #return [x for x in self  if self.node[x]['type']=='fac']
        return self.graph['facs']


    def __call__(self, fac, *vals):
        #print '%s%s' % (fac, vals)
        f = self.node[fac]['p'] #: table
        return f[vals]

    def val(self, var):
        assert self.type(var) == 'var'
        return G.node[var]['x']

    def vals(self, var):
        """
        discrete random variables
        ->
        all functions on them are arrays
        all their values are indices
        """

        assert self.type(var) == 'var'
        return range(self.node[var]['d'])

    def conditions(self, var, val):
        pass

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Functions on Factor Graphs


def isTree(G):
    return nx.cycle_basis(G) == []


# Example Factor Graphs


def factor_tree():
    """
    4 vars
    2 facs

    """

    G = FactorGraph()
    a,b,c,d = 'a','b','c','d'
    G.graph['root'] = c
    
    G.add_var(a, d=1)
    G.add_var(b, d=2)
    G.add_var(c, d=3)
    G.add_var(d, d=4)

    p1 = pd(magic(1))
    p2 = pd(magic(2))
    p4 = pd(magic(4))

    G.add_fac(p1, [a], name='f[a]')
    G.add_fac(p2, [b], name='f[b]')
    G.add_fac(p4, [d], name='f[d]')
    
    p13 = pd(magic((1,3)))
    p23 = pd(magic((2,3)))
    p34 = pd(magic((3,4)))

    G.add_fac(p13, [a,c], name='f[ac]')
    G.add_fac(p23, [b,c], name='f[bc]')
    G.add_fac(p34, [c,d], name='f[cd]')

    return G


def factor_small():
    """ small list """

    G = FactorGraph()

    a,b = 'a','b'
    G.add_var(a, d=2)
    G.add_var(b, d=2)

    fa  = pd(magic((2)))
    fab = pd(magic((2,2)))
    G.add_fac(fa,  [a],   name='f[a]')
    G.add_fac(fab, [a,b], name='f[ab]')

    return G
    

def factor_list():
    """
    4 vars
    3 fac

    """

    G = FactorGraph()

    a,b,c,d = 'a','b','c','d'
    G.add_var(a, d=1)
    G.add_var(b, d=2)
    G.add_var(c, d=3)
    G.add_var(d, d=4)

    p12 = pd(magic((1,2)))
    p23 = pd(magic((2,3)))
    p34 = pd(magic((3,4)))
    G.add_fac(p12, [a,b], name='f[ab]')
    G.add_fac(p23, [b,c], name='f[bc]')
    G.add_fac(p34, [c,d], name='f[cd]')

    return G


def factor_clique():
    """
    4 vars
    1 fac

    """

    G = FactorGraph()

    a,b,c,d = 'a','b','c','d'
    G.add_var(a, d=1)
    G.add_var(b, d=2)
    G.add_var(c, d=3)
    G.add_var(d, d=4)

    p = pd(magic((1,2,3,4)))
    G.add_fac(p, [a,b,c,d], name='f[abcd]')

    return G


def factor_3f1v():
    G = FactorGraph()
    v = 'v'
    d = 3
    G.add_var(v, d=d)

    p1 = pd(ones(d))
    p2 = pd(magic(d))
    p3 = array([0.8,0.15,0.05])
    
    G.add_fac(p1, v, name='f1')
    G.add_fac(p2, v, name='f2')
    G.add_fac(p3, v, name='f3')

    return G


def factor_1f3v():
    G = FactorGraph()

    a,b,c = 'a','b','c'
    G.add_var(a, d=2)
    G.add_var(b, d=3)
    G.add_var(c, d=4)

    f = pd(magic((2,3,4)))
    G.add_fac(f, [a, b, c], name='f')

    return G
    


T = factor_tree()

L = factor_list()

C = factor_clique()

#S = factor_star()

#print 'T =', T.node
if __name__=='__main__':
    nx.draw(T)
    show()
