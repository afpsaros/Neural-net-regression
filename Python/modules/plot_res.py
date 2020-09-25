# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:18:47 2020

@author: afpsaros
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from data_burgers import data_getter
from reg_classes import DNN

from global_search import random_global_cv

n = 200
s = 0
val_split = 0.7
normalize = 1

#%%
with open("apo_results.txt", "r") as f:
    lines = [ line.strip( ) for line in list(f) ]

nus = [float(lines[i][5:8]) for i in range(0, len(lines), 2)]

depths = [int(lines[i].split(',')[0][2:]) for i in range(1, len(lines), 2)]


widths = [lines[i].split(',')[1] for i in range(1, len(lines), 2)]

widths = [int(w.split(')')[0][1:]) for w in widths]

lrs = [float(lines[i].split(',')[4][1:-2]) for i in range(1, len(lines), 2)]
#%%
for nu_i, nu in enumerate(nus):
    print('nu = ', nu)
    
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(normalize)
    
    DNN_dict = {
        'input dimension': 2,
        'output dimension': 1,
        'number of layers': depths[nu_i],
        'layer width': widths[nu_i], 
    }
    
    fit_dict = {
        'initialize': 1,
        'wd_par': 0,
        'num_epochs': 2000,
        'Xt': data.Xt_norm,
        'Yt': data.Yt_norm,
        'Xv': data.Xv_norm,
        'Yv': data.Yv_norm,
        'lr': lrs[nu_i]
    }      

    eval_dict = {
        'Xe': data.Xe_norm,
        'Ye': data.Ye_norm
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
        print(model.score(eval_dict)[0])
        
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
    
    plt.title('x, u plots for different times')
    plt.xlabel('x')
    plt.ylabel('u')
    
    name = 'plot_nu_' + '{}'.format(nu_i) + '.png'
    plt.savefig(name, dpi = 400)
    plt.show()  
    