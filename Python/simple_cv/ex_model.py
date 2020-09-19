# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:48:24 2020

@author: afpsaros
"""

import numpy as np

class ex_model:
    
    def fit(self, fit_dict):
        self.pa = fit_dict['a']
        self.pb = fit_dict['b']
        self.pc = fit_dict['c']
        
        return self
        
    def score(self, eval_dict):
        Xe, Ye = list(eval_dict.values())
        return sum([self.pa, self.pb, self.pc])
    
    
        