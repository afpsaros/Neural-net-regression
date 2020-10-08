# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:13:10 2020

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
    
with open('ens_out.txt', 'rb') as f:
    [ms, ENS_errors] = pickle.load(f) 
    # ms = pickle.load(f) 
    
with open('snapen_out.txt', 'rb') as f:
    [budgets, B_errors, ms] = pickle.load(f)    
#%%    
for i in range(0, len(budgets)):    
    plt.scatter(budgets[0] * (i + 1) * np.ones(M), list(zip(*M_errors))[i])    

plt.plot(budgets, np.mean(list(zip(*M_errors)), 1), label = '1')  

for mi, m in enumerate(ms):
           
    plt.plot(budgets, list(zip(*B_errors))[mi], '-', label = '{}'.format(m))

plt.legend()
plt.show()  

#%%
plt.plot(ms, [np.mean(el) for el in ENS_errors], '-o', label = 'true ens')   
 
sm_mean = np.mean(list(zip(*M_errors))[-1])

plt.plot(ms, sm_mean * np.ones(len(ms)), label = 'single model')      

plt.plot(ms, B_errors[-1], '-o', label = 'cheap ens') 
plt.show()
print('rel', (sm_mean - np.mean(ENS_errors[-1])) / sm_mean * 100) 