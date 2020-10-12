# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:45:33 2020

@author: afpsaros
"""

# =============================================================================
# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 
# import reg_classes
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
        y = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3 
        
        y += rng.normal(0, self.s, [self.n, 1])
        
        data = np.concatenate((x, y), axis=1)
        
        rng.shuffle(data)
        
        self.data_tr, self.data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
        
        self.n_tr, self.n_val = self.data_tr.shape[0], self.data_val.shape[0]
        
        x = np.linspace(-1, 2, self.n_eval)
        x = x.reshape((self.n_eval, 1))
        y = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3
        
        self.data_eval = np.concatenate((x, y), axis=1)
        
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
            l2_error = np.mean(np.square(pred - self.Ye))
            l2_rel = np.sqrt(np.sum(np.square(pred - self.Ye)) / np.sum(np.square(self.Ye)))
            l1_rel = np.sum(np.abs(pred - self.Ye)) / np.sum(np.abs(self.Ye))
            l1_point = np.abs(pred - self.Ye)
            
            return l2_error, l2_rel, l1_rel, l1_point
               
if __name__ == '__main__':
    
    
    
    n = 30 
    s = 0
    val_split = 0.5
    n_eval = 50
   
    data = data_getter(n, s, val_split, n_eval).create_data()
    
    data.plot_tr_data()
    data.plot_eval_data(1)


    