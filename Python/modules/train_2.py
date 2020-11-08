# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:29:18 2020

@author: afpsaros
"""

import tensorflow as tf
from reg_classes import DNN
from callbacks import *

import matplotlib.pyplot as plt 
from data_file import data_getter

n = 30
s = 0
val_split = 0.8
n_eval = 100

scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)


DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 1,
    'layer width': 2, 
}

callbacks = []

fit_dict = {
    'callbacks': callbacks,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': 1000,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,
    'lr': 0.01,
    'decay': None,
}

sess = tf.Session()
model_1 = DNN.standard(DNN_dict, sess, seed = 1)
model_2 = DNN.standard(DNN_dict, sess, seed = 1)

model_1.hyper_initial()
model_1.fp = model_1.ffn()
err_1 = model_1.errors()[0]

model_2.hyper_initial()
model_2.fp = model_2.ffn()
err_2 = model_2.errors()[0]

init = tf.global_variables_initializer()     

sess.run(init)   
#%%
print(sess.run(model_1.weights))
print(sess.run(model_2.weights))

lr = 1e-3

obj_1 = err_1   
obj_2 = err_2
     
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
         
train_op_1 = optimizer.minimize(obj_1)
train_op_2 = optimizer.minimize(obj_2)

sess.run(tf.variables_initializer(optimizer.variables()))   
    
#%%
if scale == 1:
    x = data.Xe.reshape(-1,1)
    x_scal = data.Xe_scal
    pred = model.pred(x_scal)
    pred = data.scaler_y.inverse_transform(pred)
    plt.plot(x, pred, '.')
    data.plot_eval_data(1)
    
    print(data.assess_pred(pred)[:3])
    plt.plot(x, data.assess_pred(pred)[-1])        
    plt.show()
   
else:
    x = data.Xe.reshape(-1,1)
    pred = model.pred(x)
    plt.plot(x, pred, '.')
    data.plot_eval_data(1)     
    
    print(data.assess_pred(pred)[:3])
    plt.plot(x, data.assess_pred(pred)[-1])
    plt.show()        
