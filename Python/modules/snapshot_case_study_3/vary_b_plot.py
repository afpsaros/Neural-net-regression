# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:46:39 2020

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

import pickle 

with open('vary_b_out.txt', 'rb') as f:
    [budgets, snap_B_errors] = pickle.load(f)
    
with open('single_cycle_ens_out.txt', 'rb') as f:
    [budgets, single_cycle_B_errors] = pickle.load(f)     
    
#%%
plt.figure(figsize=(5,4))
plt.title('Snapshot ensembles vs Single-cycle ensembles')
plt.plot(budgets, snap_B_errors, '-o', label = 'Snapshot ensemble')
plt.plot(budgets, single_cycle_B_errors, '-o', label = 'Single-cycle ensemble')
plt.xlabel('Training budget (epochs)')
plt.ylabel('Test MSE')
plt.xticks(budgets)
plt.xlim([budgets[0], budgets[-1]])
plt.legend()
plt.tight_layout()
plt.savefig('vary_b.png', dpi = 300)
plt.show()
