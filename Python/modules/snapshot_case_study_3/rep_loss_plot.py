# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:53:32 2020

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
#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal
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
    'lr': None,
    'decay': None,
}

with open("sm_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    
DNN_dict['number of layers'] = int(line.split(',')[0][2:])

width = line.split(',')[1]
DNN_dict['layer width'] = int(width.split(')')[0][1:])

fit_dict['lr'] = float(line.split(',')[2][2:])
#%%
g = tf.Graph()
sess = tf.Session(graph = g)
with g.as_default() as g:    

    callbacks = []
    loss_hist = 1
    loss = None if loss_hist == 0 else Losshistory()   
    if loss is not None: callbacks.append(loss)    
    
    fit_dict['callbacks'] = callbacks
    
    # model = DNN.standard(DNN_dict, sess, seed = 0)
    model = DNN.standard(DNN_dict, sess, seed = 1)
    
    model.fit_from_dict(fit_dict)
        
tr_error, val_error = loss.get_loss_history()

plt.yscale('log')
plt.plot(tr_error, label = 'training')
plt.plot(val_error, label = 'validation')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.tight_layout()
plt.savefig('sm_rep_loss_plot.png', dpi = 300)
plt.show()
#%%
with open("snapen_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    

fit_dict['lr'] = float(line.split(',')[2][2:])
# print(lr, lr * 10)

lr_ratio = float(line.split(',')[4][1:-2])
# print(lr_ratio, lr_ratio * 10)
#%%
g = tf.Graph()
sess = tf.Session(graph = g)
with g.as_default() as g:    

    callbacks = []
    loss_hist = 1
    loss = None if loss_hist == 0 else Losshistory()   
    if loss is not None: callbacks.append(loss)   
    
    fit_dict['callbacks'] = callbacks
    
    snap_step = given_step
    fit_dict['decay'] = ['cosine_restarts',snap_step, lr_ratio, 1., 1.]
        
    # model = DNN.standard(DNN_dict, sess, seed = 0)
    model = DNN.standard(DNN_dict, sess, seed = 1)
    
    model.fit_from_dict(fit_dict)
        
tr_error, val_error = loss.get_loss_history()

plt.yscale('log')
plt.plot(tr_error, label = 'training')
plt.plot(val_error, label = 'validation')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.tight_layout()
plt.savefig('snap_rep_loss_plot.png', dpi = 300)
plt.show()