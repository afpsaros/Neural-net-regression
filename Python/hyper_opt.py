# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:17:35 2020

@author: afpsaros
"""

from collections import defaultdict
import numpy as np
from reg_classes import DNN

class grid_cv:
    
    def __init__(self, ev_params, refit, adv_refit):
        self.ev_params = ev_params
        self.keys = list(ev_params.keys())
        self.refit = refit
        self.adv_refit = adv_refit
        
    def fit(self, model, fit_dict, eval_dict):
        
        ev_fit_dict = fit_dict
        self.scores = defaultdict(float)
        
        grid = np.meshgrid(*list(self.ev_params.values()))
        n = grid[0].size
        
        self.vals = [g.reshape(n) for g in grid]
        self.grid_d = dict(zip(self.keys, self.vals))
        
        vals_vert = list(zip(*self.vals))
        for i in range(n):
            print('{} out of {}'.format(i + 1, n))
            for key in self.keys:
                ev_fit_dict[key] = self.grid_d[key][i]
              
            self.scores[vals_vert[i]] = \
            model.fit_from_dict(ev_fit_dict).score(eval_dict)[0]
        
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
         print(k, v)
         m = min(v)
         return k[v.index(m)], m      
     

 
    
    
class grid_cv_arch(grid_cv):
  
    def __init__(self, ev_params, refit, adv_refit, ev_arch):
        
        self.ev_params = ev_params
        self.cv = grid_cv(self.ev_params, 0, 0)
        
        # self.cv = grid_cv(ev_params, refit, adv_refit)
        self.keys = list(ev_arch.keys())
        self.ev_arch = ev_arch
        self.refit = refit
        self.adv_refit = adv_refit
        
    def fit(self, fit_dict, val_dict, DNN_dict, sess):
        
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
            print('{} out of {} architectures'.format(i + 1, n))
            for key in self.keys:
                ev_DNN_dict[key] = self.grid_d[key][i]
                
            self.model = DNN.standard(ev_DNN_dict, sess)
            if self.fit_dict['initialize'] == 0:
                self.model.initialize(self.fit_dict['Xt'], self.fit_dict['Yt'])
            
            # scores, best_params, min_score = cv.fit(model, fit_dict, val_dict)
        
            self.scores[vals_vert[i]], self.best_scores[vals_vert[i]] = \
                self.cv.fit(self.model, self.fit_dict, val_dict)
            
        
        self.best = self.minscore()
        
        if self.refit == 1: 
            
            for i, key in enumerate(self.keys):
                ev_DNN_dict[key] = self.best[0][i]
                print('best arch', key, ev_DNN_dict[key])
                
            self.model = DNN.standard(ev_DNN_dict, sess)
            if self.fit_dict['initialize'] == 0:
                self.model.initialize(self.fit_dict['Xt'], self.fit_dict['Yt'])            
            
            for i, key in enumerate(list(self.ev_params.keys())):
                self.fit_dict[key] = self.best[1][0][i]
                print('best params', key, self.fit_dict[key])
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
            
    # def minscore(self):
    #     self.best_scores = defaultdict(float)
    #     print(self.scores)
    #     for i, par_dict in enumerate(self.scores.values()):
    #          v = list(par_dict[1])
    #          k = list(par_dict.keys())
    #          m = min(v)
             
    #          self.best_scores[vals_vert[i]] = \
    #              (k[v.index(m)], m)
