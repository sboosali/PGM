#!/usr/bin/python
from __future__ import division
from sam.sam import *
from sam import sam

from numpy import *
import numpy as np
from matplotlib.pyplot import *
import nltk
import numpy.random as sample
import scipy.stats as pdf

from factorgraph import *

zeros = sam.splat(zeros)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# INIT

G = FactorGraph()

def add_var(G, *args, **kwargs):
    G.add_var(*args, **kwargs)

def add_fac(G, *args, **kwargs):
    G.add_fac(*args, **kwargs)

MINVOLSET = 'MINVOLSET'
add_var(G, MINVOLSET, 3)
p = [ 0.05, 0.9, 0.05 ]
add_fac(G, p, [MINVOLSET])

VENTMACH = 'VENTMACH'
add_var(G, VENTMACH, 4)
p = zeros(4,3)
p[:,0] = [ 0.05, 0.93, 0.01, 0.01 ]
p[:,1] = [ 0.05, 0.01, 0.93, 0.01 ]
p[:,2] = [ 0.05, 0.01, 0.01, 0.93 ]
add_fac(G, p, [VENTMACH, MINVOLSET])

DISCONNECT = 'DISCONNECT'
add_var(G, DISCONNECT, 2) 
p = [ 0.1, 0.9 ]
add_fac(G, p, [DISCONNECT])

# VENTUBE | VENTMACH, DISCONNECT
VENTTUBE = 'VENTTUBE'
add_var(G, VENTTUBE, 4) 
p = zeros(4,4,2)
p[:,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,3,0] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,3,1] = [ 0.01, 0.01, 0.01, 0.97 ]
add_fac(G, p, [VENTTUBE, VENTMACH, DISCONNECT])

PULMEMBOLUS = 'PULMEMBOLUS'
add_var(G, PULMEMBOLUS,2) 
p = [ 0.01, 0.99 ]
add_fac(G, p, [PULMEMBOLUS])

INTUBATION = 'INTUBATION'
add_var(G, INTUBATION,3) 
p = [ 0.92, 0.03, 0.05 ]
add_fac(G, p, [INTUBATION])

# PAP | PULMEMBOLUS
PAP = 'PAP'
add_var(G, PAP,3)
p = zeros(3,2)
p[:,0] = [ 0.01, 0.19, 0.8 ]
p[:,1] = [ 0.05, 0.9, 0.05 ]
add_fac(G, p, [PAP, PULMEMBOLUS])

# SHUNT | PULMEMBOLUS, INTUBATION
SHUNT = 'SHUNT'
add_var(G, SHUNT,2) 
p = zeros(2,2,3)
p[:,0,0] = [ 0.1, 0.9 ]
p[:,0,1] = [ 0.1, 0.9 ]
p[:,0,2] = [ 0.01, 0.99 ]
p[:,1,0] = [ 0.95, 0.05 ]
p[:,1,1] = [ 0.95, 0.05 ]
p[:,1,2] = [ 0.05, 0.95 ]
add_fac(G, p, [SHUNT, PULMEMBOLUS, INTUBATION])

KINKEDTUBE = 'KINKEDTUBE'
add_var(G, KINKEDTUBE,2)   
p = [ 0.04, 0.96]
add_fac(G, p, [KINKEDTUBE])

# PRESS | VENTTUBE, KINKEDTUBE, INTUBATION
PRESS = 'PRESS'
add_var(G, PRESS,4) 
p = zeros(4,4,2,3)
p[:,0,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,0,1] = [ 0.01, 0.3, 0.49, 0.2 ]
p[:,0,0,2] = [ 0.01, 0.01, 0.08, 0.9 ]
p[:,0,1,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,0,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1,2] = [ 0.1, 0.84, 0.05, 0.01 ]
p[:,1,0,0] = [ 0.05, 0.25, 0.25, 0.45 ]
p[:,1,0,1] = [ 0.01, 0.15, 0.25, 0.59 ]
p[:,1,0,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,1,0] = [ 0.01, 0.29, 0.3, 0.4 ]
p[:,1,1,1] = [ 0.01, 0.01, 0.08, 0.9 ]
p[:,1,1,2] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,2,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,0,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,2,0,2] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,1,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,2,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,1,2] = [ 0.4, 0.58, 0.01, 0.01 ]
p[:,3,0,0] = [ 0.2, 0.75, 0.04, 0.01 ]
p[:,3,0,1] = [ 0.2, 0.7, 0.09, 0.01 ]
p[:,3,0,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,1,0] = [ 0.010000001, 0.90000004, 0.080000006, 0.010000001 ]
p[:,3,1,1] = [ 0.01, 0.01, 0.38, 0.6 ]
p[:,3,1,2] = [ 0.01, 0.01, 0.01, 0.97 ]
add_fac(G, p, [PRESS, VENTTUBE, KINKEDTUBE, INTUBATION])

# VENTLUNG | VENTTUBE, KINKEDTUBE, INTUBATION
VENTLUNG = 'VENTLUNG'
add_var(G, VENTLUNG,4) 
p = zeros(4,4,2,3)
p[:,0,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,0,1] = [ 0.95000005, 0.030000001, 0.010000001, 0.010000001 ]
p[:,0,0,2] = [ 0.4, 0.58, 0.01, 0.01 ]
p[:,0,1,0] = [ 0.3, 0.68, 0.01, 0.01 ]
p[:,0,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,0,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,0,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,1,0] = [ 0.95000005, 0.030000001, 0.010000001, 0.010000001 ]
p[:,1,1,1] = [ 0.5, 0.48, 0.01, 0.01 ]
p[:,1,1,2] = [ 0.3, 0.68, 0.01, 0.01 ]
p[:,2,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,0,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,2,0,2] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,1,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,2,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,2,1,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,1,0] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,3,1,1] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,3,1,2] = [ 0.01, 0.01, 0.01, 0.97 ]
add_fac(G, p, [VENTLUNG, VENTTUBE, KINKEDTUBE, INTUBATION])

FIO2 = 'FIO2'
add_var(G, FIO2,2) 
p = [ 0.05, 0.95 ]
add_fac(G, p, [FIO2])

# MINVOL | VENTLUNG, INTUBATION
MINVOL = 'MINVOL'
add_var(G, MINVOL,4) 
p = zeros(4, 4, 3)
p[:,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,0,2] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,1,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,2] = [ 0.6, 0.38, 0.01, 0.01 ]
p[:,2,0] = [ 0.5, 0.48, 0.01, 0.01 ]
p[:,2,1] = [ 0.5, 0.48, 0.01, 0.01 ]
p[:,2,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,3,1] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,3,2] = [ 0.01, 0.01, 0.01, 0.97 ]
add_fac(G, p, [MINVOL, VENTLUNG, INTUBATION])

# VENTALV | VENTLUNG, INTUBATION
VENTALV = 'VENTALV'
add_var(G, VENTALV,4) 
p = zeros(4,4,3)
p[:,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,0,2] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,1,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,2] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,2,0] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,1] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,2,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0] = [ 0.030000001, 0.95000005, 0.010000001, 0.010000001 ]
p[:,3,1] = [ 0.01, 0.94, 0.04, 0.01 ]
p[:,3,2] = [ 0.01, 0.88, 0.1, 0.01 ]
add_fac(G, p, [VENTALV, VENTLUNG, INTUBATION])

ANAPHYLAXIS = 'ANAPHYLAXIS'
add_var(G, ANAPHYLAXIS,2) 
p = [ 0.01, 0.99 ]
add_fac(G, p, [ANAPHYLAXIS])

# PVSAT | VENTALV, FIO2
PVSAT = 'PVSAT'
add_var(G, PVSAT,3) 
p = zeros(3,4,2)
p[:,0,0] = [ 1.0, 0.0, 0.0 ]
p[:,0,1] = [ 0.99, 0.01, 0.0 ]
p[:,1,0] = [ 0.95, 0.04, 0.01 ]
p[:,1,1] = [ 0.95, 0.04, 0.01 ]
p[:,2,0] = [ 1.0, 0.0, 0.0 ]
p[:,2,1] = [ 0.95, 0.04, 0.01 ]
p[:,3,0] = [ 0.01, 0.95, 0.04 ]
p[:,3,1] = [ 0.01, 0.01, 0.98 ]
add_fac(G, p, [PVSAT, VENTALV, FIO2])

# ARTCO2 | VENTALV
ARTCO2 = 'ARTCO2'
add_var(G, ARTCO2,3) 
p = zeros(3,4)
p[:,0] = [ 0.01, 0.01, 0.98 ]
p[:,1] = [ 0.01, 0.01, 0.98 ]
p[:,2] = [ 0.04, 0.92, 0.04 ]
p[:,3] = [ 0.9, 0.09, 0.01 ]
add_fac(G, p, [ARTCO2, VENTALV])

# TPR | ANAPHYLAXIS
TPR = 'TPR'
add_var(G, TPR,3) 
p = zeros(3,2)
p[:,0] = [ 0.98, 0.01, 0.01 ]
p[:,1] = [ 0.3, 0.4, 0.3 ]
add_fac(G, p, [TPR, ANAPHYLAXIS])

# SAO2 | SHUNT, PVSAT
SAO2 = 'SAO2'
add_var(G, SAO2,3)   
p = zeros(3, 2, 3)
p[:,0,0] = [ 0.98, 0.01, 0.01 ]
p[:,0,1] = [ 0.01, 0.98, 0.01 ]
p[:,0,2] = [ 0.01, 0.01, 0.98 ]
p[:,1,0] = [ 0.98, 0.01, 0.01 ]
p[:,1,1] = [ 0.98, 0.01, 0.01 ]
p[:,1,2] = [ 0.69, 0.3, 0.01 ]
add_fac(G, p, [SAO2, SHUNT, PVSAT])

INSUFFANESTH = 'INSUFFANESTH'
add_var(G, INSUFFANESTH,2)       
p = [ 0.1, 0.9]
add_fac(G, p, [INSUFFANESTH])

# EXPCO2 | VENTLUNG, ARTCO2
EXPCO2 = 'EXPCO2'
add_var(G, EXPCO2,4) 
p = zeros(4,4,3)
p[:,0,0] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,0,1] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,0,2] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,1,0] = [ 0.01, 0.97, 0.01, 0.01 ]
p[:,1,1] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,1,2] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,0] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,1] = [ 0.01, 0.01, 0.97, 0.01 ]
p[:,2,2] = [ 0.97, 0.01, 0.01, 0.01 ]
p[:,3,0] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,3,1] = [ 0.01, 0.01, 0.01, 0.97 ]
p[:,3,2] = [ 0.01, 0.01, 0.01, 0.97 ]
add_fac(G, p, [EXPCO2, VENTLUNG, ARTCO2])

LVFAILURE = 'LVFAILURE'
add_var(G, LVFAILURE,2) 
p = [ 0.05, 0.95 ]
add_fac(G, p, [LVFAILURE])

HYPOVOLEMIA = 'HYPOVOLEMIA'
add_var(G, HYPOVOLEMIA,2) 
p = [ 0.2, 0.8 ]
add_fac(G, p, [HYPOVOLEMIA])

# CATECHOL | TPR, SAO2, INSUFFANESTH, ARTCO2
CATECHOL = 'CATECHOL'
add_var(G, CATECHOL,2) 
p = zeros(2,3,3,2,3)
p[:,0,0,0,0] = [ 0.01, 0.99 ]
p[:,0,0,0,1] = [ 0.01, 0.99 ]
p[:,0,0,0,2] = [ 0.01, 0.99 ]
p[:,0,0,1,0] = [ 0.01, 0.99 ]
p[:,0,0,1,1] = [ 0.01, 0.99 ]
p[:,0,0,1,2] = [ 0.01, 0.99 ]
p[:,0,1,0,0] = [ 0.01, 0.99 ]
p[:,0,1,0,1] = [ 0.01, 0.99 ]
p[:,0,1,0,2] = [ 0.01, 0.99 ]
p[:,0,1,1,0] = [ 0.01, 0.99 ]
p[:,0,1,1,1] = [ 0.01, 0.99 ]
p[:,0,1,1,2] = [ 0.01, 0.99 ]
p[:,0,2,0,0] = [ 0.01, 0.99 ]
p[:,0,2,0,1] = [ 0.01, 0.99 ]
p[:,0,2,0,2] = [ 0.01, 0.99 ]
p[:,0,2,1,0] = [ 0.05, 0.95 ]
p[:,0,2,1,1] = [ 0.05, 0.95 ]
p[:,0,2,1,2] = [ 0.01, 0.99 ]

p[:,1,0,0,0] = [ 0.01, 0.99 ]
p[:,1,0,0,1] = [ 0.01, 0.99 ]
p[:,1,0,0,2] = [ 0.01, 0.99 ]
p[:,1,0,1,0] = [ 0.05, 0.95 ]
p[:,1,0,1,1] = [ 0.05, 0.95 ]
p[:,1,0,1,2] = [ 0.01, 0.99 ]
p[:,1,1,0,0] = [ 0.05, 0.95 ]
p[:,1,1,0,1] = [ 0.05, 0.95 ]
p[:,1,1,0,2] = [ 0.01, 0.99 ]
p[:,1,1,1,0] = [ 0.05, 0.95 ]
p[:,1,1,1,1] = [ 0.05, 0.95 ]
p[:,1,1,1,2] = [ 0.01, 0.99 ]
p[:,1,2,0,0] = [ 0.05, 0.95 ]
p[:,1,2,0,1] = [ 0.05, 0.95 ]
p[:,1,2,0,2] = [ 0.01, 0.99 ]
p[:,1,2,1,0] = [ 0.05, 0.95 ]
p[:,1,2,1,1] = [ 0.05, 0.95 ]
p[:,1,2,1,2] = [ 0.01, 0.99 ]

p[:,2,0,0,0] = [ 0.7, 0.3 ]
p[:,2,0,0,1] = [ 0.7, 0.3 ]
p[:,2,0,0,2] = [ 0.1, 0.9 ]
p[:,2,0,1,0] = [ 0.7, 0.3 ]
p[:,2,0,1,1] = [ 0.7, 0.3 ]
p[:,2,0,1,2] = [ 0.1, 0.9 ]
p[:,2,1,0,0] = [ 0.7, 0.3 ]
p[:,2,1,0,1] = [ 0.7, 0.3 ]
p[:,2,1,0,2] = [ 0.1, 0.9 ]
p[:,2,1,1,0] = [ 0.95, 0.05 ]
p[:,2,1,1,1] = [ 0.99, 0.01 ]
p[:,2,1,1,2] = [ 0.3, 0.7 ]
p[:,2,2,0,0] = [ 0.95, 0.05 ]
p[:,2,2,0,1] = [ 0.99, 0.01 ]
p[:,2,2,0,2] = [ 0.3, 0.7 ]
p[:,2,2,1,0] = [ 0.95, 0.05 ]
p[:,2,2,1,1] = [ 0.99, 0.01 ]
p[:,2,2,1,2] = [ 0.3, 0.7 ]

add_fac(G, p, [CATECHOL, TPR, SAO2, INSUFFANESTH, ARTCO2])

# HISTORY | LVFAILURE
HISTORY = 'HISTORY'
add_var(G, HISTORY,2) 
p = zeros(2,2)
p[:,0] = [ 0.9, 0.1 ]
p[:,1] = [ 0.01, 0.99 ]
add_fac(G, p, [HISTORY, LVFAILURE])

# LVEDVOLUME | LVFAILURE, HYPOVOLEMIA
LVEDVOLUME = 'LVEDVOLUME'
add_var(G, LVEDVOLUME,3) 
p = zeros(3,2,2)
p[:,0,0] = [ 0.95, 0.04, 0.01 ]
p[:,0,1] = [ 0.98, 0.01, 0.01 ]
p[:,1,0] = [ 0.01, 0.09, 0.9 ]
p[:,1,1] = [ 0.05, 0.9, 0.05 ]
add_fac(G, p, [LVEDVOLUME, LVFAILURE, HYPOVOLEMIA])

# STROKEVOLUME | LVFAILURE, HYPOVOLEMIA
STROKEVOLUME = 'STROKEVOLUME'
add_var(G, STROKEVOLUME,3) 
p = zeros(3,2,2)
p[:,0,0] = [ 0.98, 0.01, 0.01 ]
p[:,0,1] = [ 0.95, 0.04, 0.01 ]
p[:,1,0] = [ 0.5, 0.49, 0.01 ]
p[:,1,1] = [ 0.05, 0.9, 0.05 ]
add_fac(G, p, [STROKEVOLUME, LVFAILURE, HYPOVOLEMIA])

ERRLOWOUTPUT = 'ERRLOWOUTPUT'
add_var(G, ERRLOWOUTPUT,2) 
p = [ 0.05, 0.95 ]
add_fac(G, p, [ERRLOWOUTPUT])

# HR | CATECHOL
HR = 'HR'
add_var(G, HR,3)
p = zeros(3, 2)
p[:,0] = [ 0.05, 0.9, 0.05 ]
p[:,1] = [ 0.01, 0.09, 0.9 ]
add_fac(G, p, [HR, CATECHOL])

ERRCAUTER = 'ERRCAUTER'
add_var(G, ERRCAUTER,2) 
p = [ 0.1, 0.9 ]
add_fac(G, p, [ERRCAUTER])

# CVP | LVEDVOLUME
CVP = 'CVP'
add_var(G, CVP,3) 
p = zeros(3,3)
p[:,0] = [ 0.95, 0.04, 0.01 ]
p[:,1] = [ 0.04, 0.95, 0.01 ]
p[:,2] = [ 0.01, 0.29, 0.7 ]
add_fac(G, p, [CVP, LVEDVOLUME])

# PCWP | LVEDVOLUME
PCWP = 'PCWP'
add_var(G, PCWP,3)     
p = zeros(3,3)
p[:,0] = [ 0.95, 0.04, 0.01 ]
p[:,1] = [ 0.04, 0.95, 0.01 ]
p[:,2] = [ 0.01, 0.04, 0.95 ]
add_fac(G, p, [PCWP, LVEDVOLUME])

# CO | STROKEVOLUME, HR
CO = 'CO'
add_var(G, CO,3)   
p = zeros(3,3,3)
p[:,0,0] = [ 0.98, 0.01, 0.01 ]
p[:,0,1] = [ 0.95, 0.04, 0.01 ]
p[:,0,2] = [ 0.8, 0.19, 0.01 ]
p[:,1,0] = [ 0.95, 0.04, 0.01 ]
p[:,1,1] = [ 0.04, 0.95, 0.01 ]
p[:,1,2] = [ 0.01, 0.04, 0.95 ]
p[:,2,0] = [ 0.3, 0.69, 0.01 ]
p[:,2,1] = [ 0.01, 0.3, 0.69 ]
p[:,2,2] = [ 0.01, 0.01, 0.98 ]
add_fac(G, p, [CO, STROKEVOLUME, HR])

# HRBP | HR, ERRLOWOUTPUT
HRBP = 'HRBP'
add_var(G, HRBP,3) 
p = zeros(3,3,2)
p[:,0,0] = [ 0.98, 0.01, 0.01 ]
p[:,0,1] = [ 0.4, 0.59, 0.01 ]
p[:,1,0] = [ 0.3, 0.4, 0.3 ]
p[:,1,1] = [ 0.98, 0.01, 0.01 ]
p[:,2,0] = [ 0.01, 0.98, 0.01 ]
p[:,2,1] = [ 0.01, 0.01, 0.98 ]
add_fac(G, p, [HRBP, HR, ERRLOWOUTPUT])

# HREKG | HR, ERRCAUTER
HREKG = 'HREKG'
add_var(G, HREKG,3) 
p = zeros(3,3,2)
p[:,0,0] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,0,1] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,1,0] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,1,1] = [ 0.98, 0.01, 0.01 ]
p[:,2,0] = [ 0.01, 0.98, 0.01 ]
p[:,2,1] = [ 0.01, 0.01, 0.98 ]
add_fac(G, p, [HREKG, HR, ERRCAUTER])

# HRSAT | HR, ERRCAUTER
HRSAT = 'HRSAT'
add_var(G, HRSAT,3)
p = zeros(3,3,2)
p[:,0,0] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,0,1] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,1,0] = [ 0.33333334, 0.33333334, 0.33333334 ]
p[:,1,1] = [ 0.98, 0.01, 0.01 ]
p[:,2,0] = [ 0.01, 0.98, 0.01 ]
p[:,2,1] = [ 0.01, 0.01, 0.98 ]
add_fac(G, p, [HRSAT, HR, ERRCAUTER])

# BP | TPR, CO
BP = 'BP'
add_var(G, BP,3)   
p = zeros(3, 3, 3)
p[:,0,0] = [ 0.98, 0.01, 0.01 ]
p[:,0,1] = [  0.98, 0.01, 0.01 ]
p[:,0,2] = [ 0.9, 0.09, 0.01 ]
p[:,1,0] = [ 0.98, 0.01, 0.01 ]
p[:,1,1] = [ 0.1, 0.85, 0.05 ]
p[:,1,2] = [ 0.05, 0.2, 0.75 ]
p[:,2,0] = [ 0.3, 0.6, 0.1 ]
p[:,2,1] = [ 0.05, 0.4, 0.55 ]
p[:,2,2] = [ 0.01, 0.09, 0.9 ]
add_fac(G, p, [BP, TPR, CO])

zeros = np.zeros

# # # # # # # # # # # # # # # # # # # # # # 
# MAIN

if __name__=='__main__':
    for n in G.node:
        print
        print '%s' % (n)
        print '%s' % (G.node[n])

    print
    print G.vals(HR)

    nx.draw(G);show()

