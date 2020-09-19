# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:17:35 2020

@author: afpsaros
"""

from collections import defaultdict
import numpy as np

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
                
        return (self.scores, *self.best) 
     
            
            
    def minscore(self):
     v = list(self.scores.values())
     k = list(self.scores.keys())
     m = min(v)
     return k[v.index(m)], m