# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:45:33 2020

@author: afpsaros
"""

import numpy as np
import matplotlib.pyplot as plt

class data_getter:
    
    def __init__(self, n, s, val_split, n_eval):
        self.n = n
        self.s = s
        self.val_split = val_split
        self.n_eval = n_eval
        
    def create_data(self):
 
        x = np.linspace(-1, 2, self.n)
        x = x.reshape((self.n, 1))
        y = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3 
        
        y += np.random.normal(0, self.s, [self.n, 1])
        
        data = np.concatenate((x, y), axis=1)
        
        np.random.shuffle(data)
        
        self.data_tr, self.data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
        
        self.n_tr, self.n_val = self.data_tr.shape[0], self.data_val.shape[0]
        
        x = np.linspace(-1, 2, self.n_eval)
        x = x.reshape((self.n_eval, 1))
        y = np.cos(x / 5)**3 + 4* np.sin(2 * x)**3 + \
        .3 * (x - 5)**2 + 0.02 * (x - 2)**3
        
        self.data_eval = np.concatenate((x, y), axis=1)
        
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
        

if __name__ == '__main__':
    
    n = 30 
    s = 0
    val_split = 0.5
    n_eval = 50
    data = data_getter(n, s, val_split, n_eval).create_data()
    
    data.plot_tr_data()
    data.plot_eval_data(1)


    