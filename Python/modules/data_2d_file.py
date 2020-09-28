# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:31:16 2020

@author: afpsaros
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from reg_classes import DNN

class data_getter:
    
    def __init__(self, n, s, val_split, n_eval):
        self.n = n
        self.s = s
        self.val_split = val_split
        self.n_eval = n_eval
        
    def create_data(self):
        rng = np.random.RandomState(1)
        
        x = np.linspace(-1, 2, self.n)
        x = x.reshape((self.n, 1))
        
        y1 = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3 
        
        y2 = np.cos(x / 5)**3 * 1000
        
        # y += np.random.normal(0, self.s, [self.n, 1])
        
        data = np.concatenate((x, y1, y2), axis=1)
        
        rng.shuffle(data)
        
        self.data_tr, self.data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
        
        self.n_tr, self.n_val = self.data_tr.shape[0], self.data_val.shape[0]
        
        x = np.linspace(-1, 2, self.n_eval)
        x = x.reshape((self.n_eval, 1))
        y1 = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3 
        
        y2 = np.cos(x / 5)**3 * 1000
        
        self.data_eval = np.concatenate((x, y1, y2), axis=1)
        
        return self        
    
    def preproc(self, scale):
        
        self.scale = scale     
        
        self.Xt, self.Yt = self.data_tr[:, [0]], self.data_tr[:, 1:3]
        self.Xv, self.Yv = self.data_val[:, [0]], self.data_val[:, 1:3]
        self.Xe, self.Ye = self.data_eval[:, [0]], self.data_eval[:, 1:3]                                        

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
    
    def plot_tr_data_1(self):
        data_tr_1 = np.concatenate((self.data_tr[:, 0].reshape(-1, 1), self.data_tr[:, 1].reshape(-1, 1)), axis=1)
        data_val_1 = np.concatenate((self.data_val[:, 0].reshape(-1, 1), self.data_val[:, 1].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_tr_1)), 'bo', label = 'train data')
        plt.plot(*list(zip(*data_val_1)), 'ro', label = 'val data')
        plt.legend()
        plt.show()

    def plot_tr_data_2(self):
        data_tr_2 = np.concatenate((self.data_tr[:, 0].reshape(-1, 1), self.data_tr[:, 2].reshape(-1, 1)), axis=1)
        data_val_2 = np.concatenate((self.data_val[:, 0].reshape(-1, 1), self.data_val[:, 2].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_tr_2)), 'bo', label = 'train data')
        plt.plot(*list(zip(*data_val_2)), 'ro', label = 'val data')
        plt.legend()
        plt.show()
        
    def plot_eval_data_1(self, show):
        data_1 = np.concatenate((self.data_eval[:, 0].reshape(-1, 1), self.data_eval[:, 1].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_1)), label = 'true function')
        
        if show == 1:
            plt.legend()
            plt.show()
            
    def plot_eval_data_2(self, show):
        data_2 = np.concatenate((self.data_eval[:, 0].reshape(-1, 1), self.data_eval[:, 2].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_2)), label = 'true function')
        
        if show == 1:
            plt.legend()
            plt.show()            
            
            
if __name__ == '__main__':
    
    n = 50 
    s = 0
    val_split = 0.7
    n_eval = 50
    
    scale = 1
   
    data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
    
    data.plot_tr_data_1()
    data.plot_tr_data_2()
    data.plot_eval_data_1(1)
    data.plot_eval_data_2(1)
    
    DNN_dict = {
        'input dimension': 1,
        'output dimension': 2,
        'number of layers': 4,
        'layer width': 50, 
    }
    
    fit_dict = {
        'initialize': 0,
        'wd_par': 0,
        'num_epochs': 5000,
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
        # print(model.score(eval_dict)[0])
           
    if scale == 1:
        x = data.Xe
        x_scal= data.Xe_scal
        pred = data.scaler_y.inverse_transform(model.pred(x_scal))
        
        plt.plot(x.reshape(-1,1), pred[:, 0], '.')
        data.plot_eval_data_1(1)    
        
        plt.plot(x.reshape(-1,1), pred[:, 1], '.')
        data.plot_eval_data_2(1)   
       
    else:
        x = data.Xe
        pred = model.pred(x)
        plt.plot(x.reshape(-1,1), pred[:, 0], '.')
        data.plot_eval_data_1(1)    
        
        plt.plot(x.reshape(-1,1), pred[:, 1], '.')
        data.plot_eval_data_2(1)   