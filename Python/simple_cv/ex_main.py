# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:54:02 2020

@author: afpsaros
"""

import numpy as np
from ex_model import ex_model
from ex_cv import grid_cv

c = 0

Xe, Ye = 0, 0
fit_dict = {
        'a': None, 
        'b': None, 
        'c': None
        }

eval_dict = {
    'Xe': Xe,
    'Ye': Ye
}

ev_params = {
        'a': np.linspace(0, 5, 6), 
        'b': [4, 5, 6], 
        'c': [7, 8, 3, 10]        
        }

model = ex_model()

cv = grid_cv(ev_params)
scores, best_params, min_score = cv.fit(model, fit_dict, eval_dict)

print(best_params, min_score)

#print(cv)