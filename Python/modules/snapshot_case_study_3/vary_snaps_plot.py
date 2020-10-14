# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:39:57 2020

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
    [budgets, M_snaps, M_preds, M_errors] = pickle.load(f)

M = len(M_snaps)  
    
with open('ens_out.txt', 'rb') as f:
    [no_models, ENS_errors] = pickle.load(f)     
    
with open('vary_snaps_out.txt', 'rb') as f:
    [snap_nums, SN_R_errors, SN_R_means] = pickle.load(f)   

with open('NC_vary_snaps_out.txt', 'rb') as f:
    [snap_nums, NC_R_errors, NC_R_means] = pickle.load(f)      
#%%  
# for i in range(len(snap_rep_errors)):          
#     plt.plot(snap_nums, snap_rep_errors[i], '-o', label = 'rep {}'.format(i+1))
plt.figure(figsize=(5,4))
# plt.title('Snapshot, no-cycle and true ensembles vs single model')
plt.plot(snap_nums, SN_R_means, '-o', label = 'Snapshot ensemble')
plt.plot(snap_nums, NC_R_means, '-o', label = 'No-cycle ensemble')    
plt.plot(no_models, [np.mean(el) for el in ENS_errors], '-o', label = 'True ensemble')   

sm_mean = np.mean(list(zip(*M_errors))[-1])

plt.plot(no_models, sm_mean * np.ones(len(no_models)), '--', label = 'Single model')      

plt.xlim([snap_nums[0], snap_nums[-1]])
plt.xlabel('Number of models', fontsize = 15)
plt.ylabel('Test error', fontsize = 15)
plt.xticks(snap_nums) 
plt.legend()
plt.tight_layout()
plt.savefig('vary_snaps.png', dpi = 300)
plt.show()
# print('rel', (sm_mean - np.mean(ENS_errors[-1])) / sm_mean * 100) 