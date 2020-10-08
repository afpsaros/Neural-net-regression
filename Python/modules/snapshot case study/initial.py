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
   
n = 200 
s = 0
val_split = 0.7
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
# data.plot_tr_data()
# data.plot_eval_data(1)
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
    'num_epochs': 15000,
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

loss_hist = 1
loss = None if loss_hist is None else Losshistory()   
if loss is not None: r_callbacks.append(loss) 

refit = 0   
n_random = 100
random_seed = 1

ev_arch = {
        'number of layers': ([4, 10], 'a'),
        'layer width': ([30, 100], 'a')
        }

ev_params = {
    'lr': ([-4, -2], 'd')
    }     
   
arch_cv = random_global_cv(ev_params, ev_arch, refit, r_callbacks, n_random, random_seed)
scores, best, model = arch_cv.fit(fit_dict, DNN_dict)

# print(arch_cv.best)

file = open("best_arch.txt","w") 
file.write('{} \n'.format(arch_cv.best[0]))  
# file.write('{} \n'.format(d))  
# file.write('{} \n'.format(w))  
file.close()    
#%%
# tr_error, val_error = loss.get_loss_history()

# plt.yscale('log')
# plt.plot(tr_error, label = 'training loss')
# plt.plot(val_error, label = 'validation error')
# plt.legend()
# plt.show() 
#%% 
         
# if scale == 1:
#     x = data.Xe.reshape(-1,1)
#     x_scal = data.Xe_scal
#     pred = data.scaler_y.inverse_transform(model.pred(x_scal))
#     plt.plot(x, pred, '.')
#     data.plot_eval_data(1)
    
#     print(data.assess_pred(pred)[0])
#     plt.plot(x, data.assess_pred(pred)[1])        

   
# else:
#     x = data.Xe.reshape(-1,1)
#     pred = model.pred(x)
#     plt.plot(x, pred, '.')
#     data.plot_eval_data(1)     
    
#     print(data.assess_pred(pred)[0])
#     plt.plot(x, data.assess_pred(pred)[1])    
#%%
# import pickle

# with open('data_instance.txt', 'wb') as f:
#     pickle.dump(data, f)
    
# #%%
# with open('data_instance.txt', 'rb') as f:
#     data_2 = pickle.load(f)

# if scale == 1:
#     x = data_2.Xe.reshape(-1,1)
#     x_scal = data_2.Xe_scal
#     pred = data_2.scaler_y.inverse_transform(model.pred(x_scal))
#     plt.plot(x, pred, '.')
#     data_2.plot_eval_data(1)
    
#     print(data_2.assess_pred(pred)[0])
#     plt.plot(x, data_2.assess_pred(pred)[1])        

   
# else:
#     x = data.Xe.reshape(-1,1)
#     pred = model.pred(x)
#     plt.plot(x, pred, '.')
#     data.plot_eval_data(1)     
    
#     print(data.assess_pred(pred)[0])
#     plt.plot(x, data.assess_pred(pred)[1])   