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
    [budgets, M_snaps, M_preds, M_errors, M_inits] = pickle.load(f)
    
with open('ca_out.txt', 'rb') as f:
    [CA_snaps, CA_preds, CA_errors, SN_R_preds] = pickle.load(f)     
    
with open('ens_out.txt', 'rb') as f:
    [no_models, ENS_errors, ENS_preds] = pickle.load(f)
    
with open('NC_vary_snaps_out.txt', 'rb') as f:
    [snap_nums, NC_R_errors, NC_R_means, NC_R_preds] = pickle.load(f)    

c = len(M_snaps[0][0])
reps = len(M_snaps) 
#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal

x = data.Xe.flatten()

#%%
r = 0

ylim = [0, 2.2]

pred = SN_R_preds[r][-1]
error = data.assess_pred(pred)[-1]
print(sum(error))

plt.plot(x, error, '-', label = 'snap ensemble')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)

plt.ylim(ylim)

plt.tight_layout()
plt.savefig('snap_rep_err.png', dpi = 300)
plt.show()

pred = pred.flatten()
pred_std = np.std(CA_preds[r], 0).flatten()
plt.plot(x, pred, '-', label = 'snap ensemble')
plt.fill_between(x, pred-2*pred_std, pred+2*pred_std)
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('snap_rep_fun.png', dpi = 300)
plt.show()
#%%
pred = M_preds[r][-1]
error = data.assess_pred(pred)[-1]
print(sum(error))

plt.plot(x, error, '-', label = 'single model')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)

plt.ylim(ylim)

plt.tight_layout()
plt.savefig('sm_rep_err.png', dpi = 300)
plt.show()

plt.plot(x, pred, '-', label = 'single model')
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('sm_rep_fun.png', dpi = 300)
plt.show()
#%%
# test_list = [[1, 4, [5,6]], [4, 6, [8,9]], [8, 3, [10,11]]]
# print(list(zip(*test_list))[-1])
# print(np.mean(list(zip(*test_list))[-1], 0))
#%%
pred = ENS_preds[-1][0]
error = data.assess_pred(pred)[-1]
print(sum(error))

plt.plot(x, error, '-', label = 'true ensemble')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)

plt.ylim(ylim)

plt.tight_layout()
plt.savefig('ens_rep_err.png', dpi = 300)
plt.show()

pred = pred.flatten()
pred_std = np.std(list(zip(*M_preds))[-1], 0).flatten()
plt.plot(x, pred, '-', label = 'true ensemble')
plt.fill_between(x, pred-2*pred_std, pred+2*pred_std)
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('ens_rep_fun.png', dpi = 300)
plt.show()
#%%
pred = NC_R_preds[r][-1]
error = data.assess_pred(pred)[-1]
print(sum(error))

plt.plot(x, error, '-', label = 'no-cycle ensemble')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)

plt.ylim(ylim)

plt.tight_layout()
plt.savefig('nc_rep_err.png', dpi = 300)
plt.show()

pred = pred.flatten()
pred_std = np.std(M_preds[r], 0).flatten()
plt.plot(x, pred, '-', label = 'no-cycle ensemble')
plt.fill_between(x, pred-2*pred_std, pred+2*pred_std)
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('nc_rep_fun.png', dpi = 300)
plt.show()







