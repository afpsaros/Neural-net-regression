# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:52:47 2020

@author: afpsaros
"""

from collections import defaultdict
import numpy as np
from reg_classes import DNN
import tensorflow as tf
import time

class random_global_cv:
    
    def __init__(self, ev_params, ev_arch, refit, adv_refit, n_random, random_seed):
        self.ev_params = ev_params
        self.ev_arch = ev_arch
        
        self.refit = refit
        self.adv_refit = adv_refit
        
        self.n_random = n_random
        self.random_seed = random_seed
        

    def fit(self, fit_dict, DNN_dict):
        
        val_dict = {
            'Xv': fit_dict['Xv'],
            'Yv': fit_dict['Yv']
        }   
        
        ev_fit_dict = fit_dict
        ev_DNN_dict = DNN_dict
        self.scores = defaultdict(float)        
        self.tocs = []       
        for i in range(self.n_random):
            tic = time.perf_counter()
            # print('{} out of {}'.format(i + 1, self.n_random))
            l1 = []
            
            for key, val in self.ev_arch.items():  
                if val[1] == 'a':
                    ev_DNN_dict[key] = int(np.random.uniform(val[0][0], val[0][1]))
                    l1.append(ev_DNN_dict[key])
                elif val[1] == 'c':
                    ev_DNN_dict[key] = int(10 ** np.random.uniform(val[0][0], val[0][1]))
                    l1.append(ev_DNN_dict[key]) 
            
            l2 = []
                
            for key, val in self.ev_params.items():  
# =============================================================================
#                     example types
#                     param_grid = {
#                         'x': ([1, 10], 'a'),
#                         'y': ([True, False], 'b'), 
#                         'z': ([np.log10(10), np.log10(50)], 'c'),
#                         'q': ([-1, -5], 'd'),
#                         's;: ([0, -1, -5], 'e')
#                         }
# =============================================================================
                if val[1] == 'a':
                    ev_fit_dict[key] = int(np.random.uniform(val[0][0], val[0][1]))
                    l2.append(ev_fit_dict[key])
                elif val[1] == 'b':
                    ev_fit_dict[key] = np.random.choice(val[0])
                    l2.append(ev_fit_dict[key])
                elif val[1] == 'c':
                    ev_fit_dict[key] = int(10 ** np.random.uniform(val[0][0], val[0][1]))
                    l2.append(ev_fit_dict[key])    
                elif val[1] == 'd':
                    ev_fit_dict[key] = 10 ** np.random.uniform(val[0][0], val[0][1])
                    l2.append(ev_fit_dict[key]) 
                elif val[1] == 'e':
                    sel = np.random.choice([0, 1])
                    if sel == 0:
                        ev_fit_dict[key] = val[0][0]
                    else: 
                        ev_fit_dict[key] = 10 ** np.random.uniform(val[0][1], val[0][2])
                    l2.append(ev_fit_dict[key])                    
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
                if self.random_seed in [0, 1]:
                    model = DNN.standard(ev_DNN_dict, sess, self.random_seed)
                else:
                    sel = np.random.choice(np.arange(1, self.random_seed + 1)) 
                    model = DNN.standard(ev_DNN_dict, sess, sel)

                self.scores[tuple([tuple(l1), tuple(l2)])] = \
                model.fit_from_dict(ev_fit_dict).score(val_dict)[0] 
                
            self.tocs.append(time.perf_counter()-tic)
            print(time.perf_counter()-tic)
            
        self.best = self.minscore()
                
        if self.refit == 1: 
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
            
                for i, key in enumerate(self.ev_arch.keys()):
                    ev_DNN_dict[key] = self.best[0][0][i]
                    # print('best arch', key, ev_DNN_dict[key])
                    
                for i, key in enumerate(self.ev_params.keys()):
                    ev_fit_dict[key] = self.best[0][1][i]       
                    # print('best params', key, ev_fit_dict[key])
                    
                if self.random_seed in [0, 1]:
                    model = DNN.standard(ev_DNN_dict, sess, self.random_seed)
                else:
                    sel = np.random.choice(np.arange(1, self.random_seed + 1)) 
                    model = DNN.standard(ev_DNN_dict, sess, sel)
                
                if self.adv_refit == 1:              
                    model.adv_fit_from_dict(ev_fit_dict)
                else: 
                    model.fit_from_dict(ev_fit_dict)
                    
        return (self.scores, self.best, model)                   
            
    def minscore(self):
         v = list(self.scores.values())
         k = list(self.scores.keys())
         # print(k, v)
         m = min(v)
         return k[v.index(m)], m   
                    
#%%
if __name__ == '__main__':  
    
    import matplotlib.pyplot as plt
    from data_burgers import data_getter
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
        'number of layers': None,
        'layer width': None, 
    }
    
    fit_dict = {
        'initialize': 1,
        'wd_par': None,
        'num_epochs': None,
        'Xt': data.Xt_norm,
        'Yt': data.Yt_norm,
        'Xv': data.Xv_norm,
        'Yv': data.Yv_norm,
        'lr': None
    }      

    refit = 1
    adv_refit = 1    
    random_sel = 1
    n_random = 50
    random_seed = 3

    ev_arch = {
            'number of layers': ([2, 8], 'a'),
            'layer width': ([10, 60], 'a')
            }
    
    ev_params = {
        'num_epochs': ([5000, 10000], 'b'),
        'wd_par': ([0, -5, -3], 'e'),
        'lr': ([-3, -2], 'd')
        }                    
 
    arch_cv = random_global_cv(ev_params, ev_arch, refit, adv_refit, n_random, random_seed)
    scores, best, model = arch_cv.fit(fit_dict, DNN_dict)
    
    print(arch_cv.best)

    
#%%
    plt.plot(arch_cv.tocs) 
    plt.show()
    
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
                
           
                
                
                
                
     