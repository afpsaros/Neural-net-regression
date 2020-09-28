# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:59:34 2020

@author: afpsaros
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from reg_classes import DNN


class data_getter:

    def create_data(self):
                    
        with open("tanh_archs.txt", "r") as f:
            lines = [ line.strip( ) for line in list(f) ]
        # print(lines)
        
        a = []
        for i in range(3):
            a.append(lines[i].split(', '))
            for j, el in enumerate(a[i]):
                if j == 0:
                    a[i][j] = int(el[1:])
                elif j == len(a[i])-1:
                    a[i][j] = int(el[:-1])
                else:
                    a[i][j] = int(el)
                    
            a[i] = np.array(a[i]).reshape(-1,1)
                    
        self.data_tr, self.data_val = np.concatenate((a[0], a[1], a[2]), axis=1), None
        
        self.n_tr, self.n_val = self.data_tr.shape[0], None
        
        self.data_eval = np.concatenate((a[0], a[1], a[2]), axis=1)
        
        return self   
    
    def preproc(self, scale):
        
        self.scale = scale     
        
        self.Xt, self.Yt = self.data_tr[:, [0]], self.data_tr[:, 1:3]
        self.Xv, self.Yv = None, None
        self.Xe, self.Ye = self.data_eval[:, [0]], self.data_eval[:, 1:3]                                        

        if self.scale == 1:
            self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_x.fit(self.Xt)
            self.scaler_y.fit(self.Yt)
            
            self.Xt_scal = self.scaler_x.transform(self.Xt)
            self.Yt_scal = self.scaler_y.transform(self.Yt)
            self.Xv_scal = None
            self.Yv_scal = None
            self.Xe_scal = self.scaler_x.transform(self.Xe)
            self.Ye_scal = self.scaler_y.transform(self.Ye)
            
        else:
            self.Xt_scal = self.Xt
            self.Yt_scal = self.Yt
            self.Xv_scal = None
            self.Yv_scal = None
            self.Xe_scal = self.Xe
            self.Ye_scal = self.Ye
            
        return self
    
    def plot_tr_data_1(self):
        data_tr_1 = np.concatenate((self.data_tr[:, 0].reshape(-1, 1), self.data_tr[:, 1].reshape(-1, 1)), axis=1)
        # data_val_1 = np.concatenate((self.data_val[:, 0].reshape(-1, 1), self.data_val[:, 1].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_tr_1)), 'bo', label = 'train data')
        # plt.plot(*list(zip(*data_val_1)), 'ro', label = 'val data')
        plt.legend()
        plt.show()

    def plot_tr_data_2(self):
        data_tr_2 = np.concatenate((self.data_tr[:, 0].reshape(-1, 1), self.data_tr[:, 2].reshape(-1, 1)), axis=1)
        # data_val_2 = np.concatenate((self.data_val[:, 0].reshape(-1, 1), self.data_val[:, 2].reshape(-1, 1)), axis=1)
        
        plt.plot(*list(zip(*data_tr_2)), 'bo', label = 'train data')
        # plt.plot(*list(zip(*data_val_2)), 'ro', label = 'val data')
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
    
    # n = 50 
    # s = 0
    # val_split = 0.7
    # n_eval = 50
    
    scale = 1
   
    data = data_getter().create_data().preproc(scale)
    
    data.plot_tr_data_1()
    data.plot_tr_data_2()
    data.plot_eval_data_1(1)
    data.plot_eval_data_2(1)
#%%    
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