# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:40:28 2020

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
   
n = 100
s = 0
val_split = 0.7
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)

with open('data_instance.txt', 'wb') as f:
    pickle.dump(data, f)
#%%
with open('budget_info.txt', 'r') as f:
    l = list(f)
    line = l[0].strip()   
    
total_budget, mini_budget, given_step = [int(line.split(',')[i]) for i in range(3)]
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': total_budget,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,
    'lr': None
}

r_callbacks = []

snap_step = None
snap = None if snap_step is None else Snapper(snap_step)   
if snap is not None: r_callbacks.append(snap) 

loss_hist = 0
loss = None if loss_hist is None else Losshistory()   
if loss is not None: r_callbacks.append(loss) 

refit = 0   
n_random = 50
random_seed = 1

ev_arch = {
        'number of layers': ([3, 8], 'a'),
        'layer width': ([20, 80], 'a')
        }

ev_params = {
    'lr': ([-4, -3], 'd')
    }     
   
arch_cv = random_global_cv(ev_params, ev_arch, refit, r_callbacks, n_random, random_seed)
scores, best, model = arch_cv.fit(fit_dict, DNN_dict)

print(arch_cv.best)

file = open("sm_best_arch.txt","w") 
file.write('{} \n'.format(arch_cv.best[0]))  
file.close()    
#%%
# with open("sm_best_arch.txt", "r") as f:

#     l = list(f)
#     line = l[0].strip()
    
# depth = int(line.split(',')[0][2:])

# print(depth)

# width = line.split(',')[1]
# width = int(width.split(')')[0][1:])

# print(width)

# lr = float(line.split(',')[2][2:])
# print(lr, lr * 10)
