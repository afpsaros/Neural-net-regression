# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:45:40 2020

@author: afpsaros
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from math import pi as PI
from math import exp as exp

from mpl_toolkits.mplot3d import Axes3D

class data_getter:
    
    def __init__(self, n, s, val_split, nu):
        self.n = n
        self.s = s
        self.val_split = val_split
        self.nu = nu
        
    
    def analytical_solution(self, NT = 150, NX = 151, TMAX = 0.5, XMAX = 2 * np.pi):
        """
        Returns the velocity field and distance for the analytical solution
        """
        NU = self.nu
        # Increments
        DT = TMAX/(NT-1)
        DX = XMAX/(NX-1)
        
        # Initialise data structures
        
        u_analytical = np.zeros((NX,NT))
        x = np.zeros(NX)
        time = []
        
        # Distance
        for i in range(0,NX):
            x[i] = i*DX
        
        # Analytical Solution
        for n in range(0,NT):
            t = n*DT
            time.append(t)
        
            for i in range(0,NX):
                phi = exp( -(x[i]-4*t)**2/(4*NU*(t+1)) ) + exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) )
        
                dphi = ( -0.5*(x[i]-4*t)/(NU*(t+1))*exp( -(x[i]-4*t)**2/(4*NU*(t+1)) )
                   -0.5*(x[i]-4*t-2*PI)/(NU*(t+1))*exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) ) )
        
                u_analytical[i,n] = -2*NU*(dphi/phi) + 4
                 
        self.u, self.x, self.t = u_analytical, x, np.array(time)
    
    def create_data_time(self):
        self.analytical_solution()        
        
        xx = self.x.reshape(-1,1)
        
        self.data_tr, self.data_val, self.data_eval = [], [], []
        for i in range(self.n_time):
            uu = self.u[:, i * 60].reshape(-1,1)
            data = np.concatenate((xx, uu), axis=1)
            self.data_eval.append(data)
            
            random_indices = np.random.choice(data.shape[0], size=self.n, replace=False)
            
            data = data[random_indices, :]
           
            _data_tr, _data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
            self.data_tr.append(_data_tr)
            self.data_val.append(_data_val)
            
        self.n_tr, self.n_val = _data_tr.shape[0], _data_val.shape[0]

        return self
    
    def create_tr_data_3D(self):
        self.analytical_solution()        
        
        self.xx = self.x.reshape(-1,1)
        self.tt = self.t.reshape(-1,1)
        self.xs = self.xx.shape[0]
        self.ts = self.tt.shape[0]
        
        ttt, xxx = np.meshgrid(self.tt, self.xx)

        ttt = np.transpose(ttt).reshape(self.ts * self.xs, 1)
        xxx = np.transpose(xxx).reshape(self.ts * self.xs, 1)  

        data = self.u[:, 0].reshape(-1,1)

        for i in range(1, self.ts):
            data = np.concatenate((data, self.u[:, i].reshape(-1,1)), axis=0)
            
        data = np.concatenate((xxx, ttt, data), axis=1)

        random_indices = np.random.choice(data.shape[0], size=self.n, replace=False)
            
        data = data[random_indices, :]

        _data_tr, _data_val = np.split(data, [int(self.val_split * self.n)], axis = 0)
        self.data_tr = _data_tr
        self.data_val = _data_val
            
        self.n_tr, self.n_val = _data_tr.shape[0], _data_val.shape[0]
        
        return self
        
    def create_eval_data_3D(self, nt_eval):   
        self.nt_eval = nt_eval
        self.data_eval = self.u[:, 0].reshape(-1,1)
        _xxx = self.xx
        _ttt = self.tt[0]*np.ones(self.xs).reshape(-1,1)
        
        self.times = np.arange(0, self.ts, self.ts//self.nt_eval)
        for i in self.times[1:]:
            _xxx = np.concatenate((_xxx, self.xx), axis=0)
            _ttt = np.concatenate((_ttt, self.tt[i]*np.ones(self.xs).reshape(-1,1)), axis=0)
            self.data_eval = np.concatenate((self.data_eval, self.u[:, i].reshape(-1,1)), axis=0)
        self.data_eval = np.concatenate((_xxx, _ttt, self.data_eval), axis=1)
                
        return self

    def plot_tr_data(self):
        for i in range(self.n_time):
            plt.plot(*list(zip(*self.data_tr[i])), 'bo', label = 'train data')
            plt.plot(*list(zip(*self.data_val[i])), 'ro', label = 'val data')
        plt.legend()
        plt.show()
        
    def plot_eval_data(self, show):
        for i in range(self.n_time):
            plt.plot(*list(zip(*self.data_eval[i])), label = 'true function')
        
        if show == 1:
            plt.legend()
            plt.show()
            
    def plot3D_train(self):
        fig = plt.figure()
        ax = Axes3D(fig)
 
        ax.scatter(*list(zip(*self.data_tr)))
        ax.scatter(*list(zip(*self.data_val)))
        ax.view_init(elev=90., azim=90)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('v')
        plt.show()

    def plot3D_eval(self, show):
        fig = plt.figure()
        ax = Axes3D(fig)
 
        ax.scatter(*list(zip(*self.data_eval)))

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('v')
        
        if show == 1:
            plt.show()
            
    def plot2D_eval(self, show):

        for i in range(len(self.times)+1):
            plt.plot(self.data_eval[i * self.xs:self.xs + i * self.xs, 0], \
                     self.data_eval[i * self.xs:self.xs + i * self.xs, 2])

        if show == 1:
            plt.show()
            
    def preproc(self, normalize):
        self.normalize = normalize          
                           
        self.Xt, self.Yt = self.data_tr[:, 0:2], self.data_tr[:, [2]]
        self.Xv, self.Yv = self.data_val[:, 0:2], self.data_val[:, [2]]
        self.Xe, self.Ye = self.data_eval[:, 0:2], self.data_eval[:, [2]]
        if self.normalize == 1:
            self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_x.fit(self.Xt)
            self.scaler_y.fit(self.Yt)
            
            self.Xt_norm = self.scaler_x.transform(self.Xt)
            self.Yt_norm = self.scaler_y.transform(self.Yt)
            self.Xv_norm = self.scaler_x.transform(self.Xv)
            self.Yv_norm = self.scaler_y.transform(self.Yv)
            self.Xe_norm = self.scaler_x.transform(self.Xe)
            self.Ye_norm = self.scaler_y.transform(self.Ye)
            
        else:
            self.Xt_norm = self.Xt
            self.Yt_norm = self.Yt
            self.Xv_norm = self.Xv
            self.Yv_norm = self.Yv
            self.Xe_norm = self.Xe
            self.Ye_norm = self.Ye
            
        return self        
#%%
if __name__ == '__main__':
    import tensorflow as tf
    from reg_classes import DNN
    
    n = 200
    s = 0
    val_split = 0.7
    nu = .1
    normalize = 1
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(normalize)
    data.plot3D_train()
    data.plot3D_eval(1)
    
    DNN_dict = {
        'input dimension': 2,
        'output dimension': 1,
        'number of layers': 4,
        'layer width': 50, 
    }
    
    fit_dict = {
        'initialize': 0,
        'wd_par': 0,
        'num_epochs': 1000,
        'Xt': data.Xt_norm,
        'Yt': data.Yt_norm,
        'Xv': data.Xv_norm,
        'Yv': data.Yv_norm,
        'lr': 0.001
    }
    
    sess = tf.Session()
    
    model = DNN.standard(DNN_dict, sess)
    model.initialize(fit_dict['Xt'], fit_dict['Yt'])
    model.fit_from_dict(fit_dict)

    data.create_eval_data_3D(nt_eval = 5).preproc(normalize)
    
    xlen = data.xs
    
    data.plot2D_eval(0)
    x = data.Xe[:xlen, 0]
    if normalize == 1:
        for i in range(len(data.times)):         
            pred = data.scaler_y.inverse_transform(model.pred(data.Xe_norm[i * xlen:xlen + i * xlen, 0:2]))
            plt.plot(x.reshape(-1,1), pred, '.')
    else:
        for i in range(len(data.times)):
            pred = model.pred(data.Xe_norm[i * xlen:xlen + i * xlen, 0:2])
            plt.plot(x, pred, '.')    
    plt.show()   
    
    
    
    
    
    
    
  


    