# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:03:47 2020

@author: afpsaros
"""


import tensorflow as tf
from reg_classes import DNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class data_getter:
    
    def __init__(self, n, s, val_split, n_eval):
        self.n = n
        self.s = s
        self.val_split = val_split
        self.n_eval = n_eval
        
    def create_data(self, nu):
        rng = np.random.RandomState(1234)
        
        x = np.linspace(-1, 1, self.n_eval).reshape(-1, 1)
        
        # y = 0.1 * np.sin(4*np.pi*x) + np.tanh(20*x)
        
        y = np.tanh(nu * x)
                
        self.data_eval = np.concatenate((x, y), axis=1)
        
        random_indices = rng.choice(self.n_eval, size=self.n, replace=False)
        
        data = self.data_eval[random_indices, :]
                
        _data_tr, _data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
        self.data_tr = _data_tr
        self.data_val = _data_val
            
        self.n_tr, self.n_val = _data_tr.shape[0], _data_val.shape[0]
        
        return self
    
    def preproc(self, scale):
        self.scale = scale       
                           
        self.Xt, self.Yt = self.data_tr[:, [0]], self.data_tr[:, [1]]
        self.Xv, self.Yv = self.data_val[:, [0]], self.data_val[:, [1]]
        self.Xe, self.Ye = self.data_eval[:, [0]], self.data_eval[:, [1]]
        if self.scale == 1:
            self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_x.fit(self.Xt)
            self.scaler_y.fit(self.Yt)
            
            self.Xt_scal = self.scaler_x.transform(self.Xt)
            self.Yt_scal = self.scaler_y.transform(self.Yt)
            self.Xv_scal = self.scaler_x.transform(self.Xv)
            self.Yv_scal = self.scaler_y.transform(self.Yv)
            self.Xe_scal = self.scaler_x.transform(self.Xe)
            self.Ye_scal = self.scaler_y.transform(self.Ye)
            
        else:
            self.Xt_scal = self.Xt
            self.Yt_scal = self.Yt
            self.Xv_scal = self.Xv
            self.Yv_scal = self.Yv
            self.Xe_scal = self.Xe
            self.Ye_scal = self.Ye
            
        return self
              
    def plot_tr_data(self):
        plt.plot(*list(zip(*self.data_tr)), 'bo', label = 'train data')
        plt.plot(*list(zip(*self.data_val)), 'ro', label = 'val data')
        plt.legend()
        plt.show()
        
    def plot_eval_data(self, show):
        plt.plot(*list(zip(*self.data_eval)), label = 'true function')
        
        if show == 1:
            plt.legend()
            plt.show()
            
    def assess_pred(self, pred):
            error = np.mean(np.square(pred - self.Ye))
            error_p = np.abs(pred - self.Ye)
            return error, error_p
               
if __name__ == '__main__':
    
    n = 100
    s = 0
    val_split = 0.8
    n_eval = 200
    
    nu = 20
      
    scale = 1
    
    data = data_getter(n, s, val_split, n_eval).create_data(nu).preproc(scale)
    data.plot_tr_data()
    data.plot_eval_data(1)
    
#%%
    DNN_dict = {
        'input dimension': 1,
        'output dimension': 1,
        'number of layers': 3,
        'layer width': 50, 
    }
    
    fit_dict = {
        'initialize': 0,
        'wd_par': 0,
        'num_epochs': 10000,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
        'Xv': data.Xv_scal,
        'Yv': data.Yv_scal,
        'lr': 0.001
    }

    eval_dict = {
        'Xe': data.Xe_scal,
        'Ye': data.Ye_scal
    }      
    
    sess = tf.Session()
    model = DNN.standard(DNN_dict, sess, seed = 1)
    model.initialize(fit_dict['Xt'], fit_dict['Yt'])
    advanced_fit = 0
    
    if advanced_fit == 1:
        model.adv_fit_from_dict(fit_dict)
    
        plt.yscale('log')
        plt.plot(model.tr_error, label = 'training loss')
        plt.plot(model.val_error, label = 'validation error')
        plt.legend()
        plt.show()
    else:
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