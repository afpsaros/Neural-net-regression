# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:22:47 2020

@author: afpsaros
"""


import sys
sys.path.insert(0,'..')

from reg_classes import DNN
from data_disc import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 
from global_search import *
from planes_projections import planes_projections

import pickle 

with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_preds, M_errors] = pickle.load(f)
    
with open('ca_out.txt', 'rb') as f:
    [CA_snaps, CA_preds, CA_errors] = pickle.load(f)     

c = len(M_snaps[0][0])
reps = len(M_snaps) 
#%%
a = [[i for _ in range(5)] for i in range(10)]
aa = zip(*a)

for r in range(reps):
    for ci in range(c):
        M_preds[r][ci] = M_preds[r][ci].flatten()
        CA_preds[r][ci] = CA_preds[r][ci].flatten()
        
plt.plot(np.mean([np.std(M_preds[r], 0) for r in range(reps)], 0))
plt.plot(np.mean([np.std(CA_preds[r], 0) for r in range(reps)], 0))
plt.show()

print(np.mean([np.std(M_preds[r], 0) for r in range(reps)]))
print(np.mean([np.std(CA_preds[r], 0) for r in range(reps)]))