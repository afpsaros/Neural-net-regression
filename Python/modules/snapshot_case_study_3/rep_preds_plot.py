# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:38:18 2020

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
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal

x = data.Xe.reshape(-1,1)
#%%
pred = CA_preds[0][-1]
plt.plot(x, pred, '.', label = 'snap ensemble')
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('snap_rep_fun.png', dpi = 300)
plt.show()

pred = M_preds[0][-1]
plt.plot(x, pred, '.', label = 'single model')
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('sm_rep_fun.png', dpi = 300)
plt.show()
#%%
pred = CA_preds[0][-1]
error = data.assess_pred(pred)[-1]
plt.plot(x, error, '-', label = 'snap ensemble')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)
plt.tight_layout()
plt.savefig('snap_rep_err.png', dpi = 300)
plt.show()

pred = M_preds[0][-1]
error = data.assess_pred(pred)[-1]
plt.plot(x, error, '-', label = 'single model')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)
plt.tight_layout()
plt.savefig('sm_rep_err.png', dpi = 300)
plt.show()