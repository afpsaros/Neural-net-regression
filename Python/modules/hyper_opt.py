# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:17:35 2020

@author: afpsaros
"""

from collections import defaultdict
import numpy as np
from reg_classes import DNN
import tensorflow as tf

import time

class param_cv:
    
    def __init__(self, ev_params, refit, adv_refit, random_sel = 0, n_random = 0):
        self.ev_params = ev_params
        self.keys = list(ev_params.keys())
        self.refit = refit
        self.adv_refit = adv_refit
        self.random_sel = random_sel
        self.n_random = n_random
        
    def fit(self, model, fit_dict):
        
        val_dict = {
            'Xv': fit_dict['Xv'],
            'Yv': fit_dict['Yv']
        }   
        
        ev_fit_dict = fit_dict
        self.scores = defaultdict(float)
        
        if self.random_sel == 1:
            for i in range(self.n_random):
                print('{} out of {}'.format(i + 1, self.n_random))
                l = []
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
                        l.append(ev_fit_dict[key])
                    elif val[1] == 'b':
                        ev_fit_dict[key] = np.random.choice(val[0])
                        l.append(ev_fit_dict[key])
                    elif val[1] == 'c':
                        ev_fit_dict[key] = int(10 ** np.random.uniform(val[0][0], val[0][1]))
                        l.append(ev_fit_dict[key])    
                    elif val[1] == 'd':
                        ev_fit_dict[key] = 10 ** np.random.uniform(val[0][0], val[0][1])
                        l.append(ev_fit_dict[key]) 
                    elif val[1] == 'e':
                        sel = np.random.choice([0, 1])
                        if sel == 0:
                            ev_fit_dict[key] = val[0][0]
                        else: 
                            ev_fit_dict[key] = 10 ** np.random.uniform(val[0][1], val[0][2])
                        l.append(ev_fit_dict[key]) 
                # print(tuple(l))
                self.scores[tuple(l)] = \
                    model.fit_from_dict(ev_fit_dict).score(val_dict)[0]          
        else:
            grid = np.meshgrid(*list(self.ev_params.values()))
            n = grid[0].size
                    
            self.vals = [g.reshape(n) for g in grid]
    
            self.grid_d = dict(zip(self.keys, self.vals))      
            vals_vert = list(zip(*self.vals))
    
            for i in range(n):
# =============================================================================
#                 print('{} out of {}'.format(i + 1, n))
# =============================================================================
                for key in self.keys:
                    ev_fit_dict[key] = self.grid_d[key][i]
                  
                self.scores[vals_vert[i]] = \
                model.fit_from_dict(ev_fit_dict).score(val_dict)[0]
        
        self.best = self.minscore()
        if self.refit == 1: 
            
            for i, key in enumerate(self.keys):
                ev_fit_dict[key] = self.best[0][i]
            if self.adv_refit == 1:              
                self.best_fit = model.adv_fit_from_dict(ev_fit_dict)
            else: 
                self.best_fit = model.fit_from_dict(ev_fit_dict)
                
        return (self.scores, self.best) 
    

    def minscore(self):
         v = list(self.scores.values())
         k = list(self.scores.keys())
         # print(k, v)
         m = min(v)
         return k[v.index(m)], m      
         
class grid_cv_arch(param_cv):
  
    def __init__(self, ev_params, refit, adv_refit, ev_arch, random_sel, n_random):
        
        self.ev_params = ev_params
        self.cv = param_cv(self.ev_params, 0, 0, random_sel, n_random)
        
        self.keys = list(ev_arch.keys())
        self.ev_arch = ev_arch
        self.refit = refit
        self.adv_refit = adv_refit
        
    def fit(self, fit_dict, DNN_dict, seed):
                
        self.fit_dict = fit_dict
        
        ev_DNN_dict = DNN_dict
        self.scores = defaultdict(float)
        self.best_scores = defaultdict(float)
        grid = np.meshgrid(*list(self.ev_arch.values()))
        n = grid[0].size

        self.vals = [g.reshape(n) for g in grid]
        self.grid_d = dict(zip(self.keys, self.vals))       
        
        
        vals_vert = list(zip(*self.vals))

        for i in range(n):
# =============================================================================
#             print('{} out of {} architectures'.format(i + 1, n))
# =============================================================================
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
                
                for key in self.keys:
                    ev_DNN_dict[key] = self.grid_d[key][i]
                    
                self.model = DNN.standard(ev_DNN_dict, sess, seed)
                if self.fit_dict['initialize'] == 0:
                    self.model.initialize(self.fit_dict['Xt'], self.fit_dict['Yt'])
                
                # scores, best_params, min_score = cv.fit(model, fit_dict, val_dict)
            
                self.scores[vals_vert[i]], self.best_scores[vals_vert[i]] = \
                    self.cv.fit(self.model, self.fit_dict)
        
        
        self.best = self.minscore()
        
        if self.refit == 1: 
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
            
                for i, key in enumerate(self.keys):
                    ev_DNN_dict[key] = self.best[0][i]
# =============================================================================
#                     print('best arch', key, ev_DNN_dict[key])
# =============================================================================
                    
                self.model = DNN.standard(ev_DNN_dict, sess, seed)
                if self.fit_dict['initialize'] == 0:
                    self.model.initialize(self.fit_dict['Xt'], self.fit_dict['Yt'])            
                
                for i, key in enumerate(list(self.ev_params.keys())):
                    self.fit_dict[key] = self.best[1][0][i]
# =============================================================================
#                     print('best params', key, self.fit_dict[key])
# =============================================================================
                if self.adv_refit == 1:              
                    self.best_fit = self.model.adv_fit_from_dict(self.fit_dict)
                else: 
                    self.best_fit = self.model.fit_from_dict(self.fit_dict)
                
        return (self.scores, self.best, self.model) 
     
            
    def minscore(self):
         v = list(self.best_scores.values())
         k = list(self.best_scores.keys())
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
    scale = 1
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(scale)
    data.plot3D_train()
    data.plot3D_eval(1)
#%%    
    DNN_dict = {
        'input dimension': 2,
        'output dimension': 1,
        'number of layers': 4,
        'layer width': 50, 
    }
    
    fit_dict = {
        'initialize': 0,
        'wd_par': None,
        'num_epochs': None,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
        'Xv': data.Xv_scal,
        'Yv': data.Yv_scal,
        'lr': None
    }
    
    # val_dict = {
    #     'Xv': data.Xv_norm,
    #     'Yv': data.Yv_norm
    # }   
  
    refit = 1
    adv_refit = 1    
    random_sel = 1
    n_random = 3
    
    if random_sel == 1:
        ev_params = {
            'num_epochs': ([1000, 1500], 'b'),
            'wd_par': ([0, -5, -3], 'e'),
            'lr': ([-3, -2], 'd')
            }
    else: 
        ev_params = {
                'num_epochs': [1000, 1500],
                'wd_par': np.linspace(1e-6, 1e-4, 3),
                'lr': np.linspace(1e-3, 1e-2, 3)
                }
        
#%%    
    sess = tf.Session()

    model = DNN.standard(DNN_dict, sess, seed = 1)
    if fit_dict['initialize'] == 0:
        model.initialize(fit_dict['Xt'], fit_dict['Yt'])
    cv = param_cv(ev_params, refit, adv_refit, random_sel, n_random)
    scores, (best_params, min_score) = cv.fit(model, fit_dict)
    
    print(best_params)
    print(min_score)
    
#%%
    data.create_eval_data_3D(nt_eval = 5).preproc(scale)
    
    xlen = data.xs
    
    data.plot2D_eval(0)
    x = data.Xe[:xlen, 0]
    if scale == 1:
        for i in range(len(data.times)):
            
            pred = data.scaler_y.inverse_transform(model.pred(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2]))
            plt.plot(x.reshape(-1,1), pred, '.')
    else:
        for i in range(len(data.times)):
            pred = model.pred(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2])
            plt.plot(x, pred, '.')
    
    plt.show()    
    
#%%
    ev_arch = {
            'number of layers': [2, 4],
            'layer width': [10, 30]
            }

    refit = 1
    adv_refit = 1    
    random_sel = 1
    n_random = 2
    
    if random_sel == 1:
        ev_params = {
            'num_epochs': ([3000, 4000], 'b'),
            'wd_par': ([0, -5, -3], 'e'),
            'lr': ([-3, -2], 'd')
            }
    else: 
        ev_params = {
                'num_epochs': [3000, 4000],
                'wd_par': np.linspace(1e-6, 1e-4, 3),
                'lr': np.linspace(1e-3, 1e-2, 3)
                }
    
    arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch, random_sel, n_random)
    tic = time.perf_counter()
    scores, best, model = arch_cv.fit(fit_dict, DNN_dict, seed = 0)    
    print(time.perf_counter() - tic)
#%%
    data.create_eval_data_3D(nt_eval = 5).preproc(scale)
    
    xlen = data.xs
    
    data.plot2D_eval(0)
    x = data.Xe[:xlen, 0]
    if scale == 1:
        for i in range(len(data.times)):
            
            pred = data.scaler_y.inverse_transform(model.pred(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2]))
            plt.plot(x.reshape(-1,1), pred, '.')
    else:
        for i in range(len(data.times)):
            pred = model.pred(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2])
            plt.plot(x, pred, '.')
    
    plt.show()    
        
