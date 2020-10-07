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
   
n = 100 
s = 0
val_split = 0.8
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 3,
    'layer width': 50, 
}

callbacks = []

snap_step = 1000
snap = None if snap_step is None else Snapper(snap_step)   
if snap is not None: callbacks.append(snap) 

loss_hist = 1
loss = None if loss_hist == 0 else Losshistory()   
if loss is not None: callbacks.append(loss) 

fit_dict = {
    'callbacks': callbacks,
    'initialize': 0,
    'wd_par': 0,
    'num_epochs': 10000,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,
    'lr': 0.01
}

eval_dict = {
    'Xe': data.Xe_scal,
    'Ye': data.Ye_scal
}      

sess = tf.Session()
model = DNN.standard(DNN_dict, sess, seed = 1)
model.initialize(fit_dict['Xt'], fit_dict['Yt'])

model.fit_from_dict(fit_dict)    
#%%           
if scale == 1:
    x = data.Xe.reshape(-1,1)
    x_scal = data.Xe_scal
    pred = data.scaler_y.inverse_transform(model.pred(x_scal))
    plt.plot(x, pred, '.')
    data.plot_eval_data(1)
    
    print(data.assess_pred(pred)[0])
    plt.plot(x, data.assess_pred(pred)[1])        

   
else:
    x = data.Xe.reshape(-1,1)
    pred = model.pred(x)
    plt.plot(x, pred, '.')
    data.plot_eval_data(1)     
    
    print(data.assess_pred(pred)[0])
    plt.plot(x, data.assess_pred(pred)[1])    



