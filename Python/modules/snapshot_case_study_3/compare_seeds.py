# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:45:03 2020

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

with open('sm_out_1.txt', 'rb') as f:
    [budgets, M_snaps_1, M_preds_1, M_errors_1] = pickle.load(f)
    
with open('sm_out_2.txt', 'rb') as f:
    [budgets, M_snaps_2, M_preds_2, M_errors_2] = pickle.load(f)
    

print([x-y for i, j in zip(M_errors_1, M_errors_2) for x, y in zip(i,j)])
print(sum([x.flatten()-y.flatten() for i, j in zip(M_preds_1, M_preds_2) for x, y in zip(i,j)]))
#%%
with open('ca_out_1.txt', 'rb') as f:
    [CA_snaps_1, CA_preds_1, CA_errors_1] = pickle.load(f)
    
with open('ca_out_2.txt', 'rb') as f:
    [CA_snaps_2, CA_preds_2, CA_errors_2] = pickle.load(f)
    

print([x-y for i, j in zip(CA_errors_1, CA_errors_2) for x, y in zip(i,j)])
print(sum([x.flatten()-y.flatten() for i, j in zip(CA_preds_1, CA_preds_2) for x, y in zip(i,j)]))
