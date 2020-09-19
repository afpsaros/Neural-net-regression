# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:56:31 2020

@author: afpsaros
"""
from collections import defaultdict
import numpy as np

class grid_cv:
    
    def __init__(self, ev_params):
        self.ev_params = ev_params
        self.keys = list(ev_params.keys())
        
    def fit(self, model, fit_dict, eval_dict):
        
        ev_fit_dict = fit_dict
        self.scores = defaultdict(float)
        
        grid = np.meshgrid(*list(self.ev_params.values()))
        n = grid[0].size
        
        self.vals = [g.reshape(n) for g in grid]
        self.grid_d = dict(zip(self.keys, self.vals))
        
        vals_vert = list(zip(*self.vals))
        for i in range(n):
            for key in self.keys:
                ev_fit_dict[key] = self.grid_d[key][i]
              
            self.scores[vals_vert[i]] = \
            model.fit(ev_fit_dict).score(eval_dict)
                
        return (self.scores, *self.minscore()) 
            
    def minscore(self):
     v = list(self.scores.values())
     k = list(self.scores.keys())
     m = min(v)
     return k[v.index(m)], m
            
            
        
                
                
                
                
        
        
        