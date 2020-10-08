# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:53:42 2020

@author: afpsaros
"""


from reg_classes import DNN
from data_file import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 

n = 50 
s = 0
val_split = 0.7
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)

x_scal = data.Xe_scal

#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 2,
    'layer width': 20, 
}

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': 1000,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': 1e-3,
    'decay': ['cosine_restarts',snap_step, 0.01, 1., 1.],
}
#%%

sess = tf.Session()

snap_step = 100
callbacks = []
snap = None if snap_step is None else Snapper(snap_step)   
if snap is not None: callbacks.append(snap) 

fit_dict['callbacks'] = callbacks

model = DNN.standard(DNN_dict, sess, seed = 0)

model.fit_from_dict(fit_dict)

snap_weights, snap_biases = snap.get_snaps()
#%%
ensemble = model.fun_ensemble(snap_weights, snap_biases)
pred = model.pred_ens(x_scal, ensemble)
pred = data.scaler_y.inverse_transform(pred)
print(data.assess_pred(pred)[0])

eval_dict = {
    'Xe': data.Xe_scal,
    'Ye': data.Ye_scal
}  

print(model.score_ens(eval_dict, ensemble))



