# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:12:52 2020

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

with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_errors] = pickle.load(f)

# M = len(M_snaps) 

with open('vary_ac_out.txt', 'rb') as f:
    [cycles, C_errors] = pickle.load(f)    

sm_mean = np.mean(list(zip(*M_errors))[-1])
#%%
plt.title('Snapshot ensembles vs Single model')
plt.plot(cycles, C_errors, '-o', label = 'Snapshot ensemble') 
plt.plot(cycles, sm_mean * np.ones(len(cycles)), '--', label = 'Single model')      
plt.xlabel('Number of cycles')
plt.ylabel('Test MSE')
plt.legend()
plt.xlim([cycles[0], cycles[-1]])
# plt.ylim([0.8*min(C_errors), 1.2 * sm_mean])
plt.xticks(ms)
plt.show()