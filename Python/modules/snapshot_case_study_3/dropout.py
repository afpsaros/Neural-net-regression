# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:02:06 2020

@author: afpsaros
"""


import sys
sys.path.insert(0,'..')

from reg_classes import DNN
from data_disc import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 
from global_search import *
from planes_projections import planes_projections

import pickle 

with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_preds, M_errors, M_inits] = pickle.load(f)
    
c = len(M_snaps[0][0])
reps = len(M_snaps) 

#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal

#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

with open("sm_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    
DNN_dict['number of layers'] = int(line.split(',')[0][2:])

width = line.split(',')[1]
DNN_dict['layer width'] = int(width.split(')')[0][1:])

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': None,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': None,
    'decay': None,
}

#%%
veclen = len(pj.abtovec(M_snaps[0][0][-1], M_snaps[0][1][-1]))

pj = planes_projections(*M_snaps[0])    

T = 100
p = 0.001
pnum = int(veclen*p)

DR_preds = []
DR_preds_mean = []
DR_preds_std = []
DR_errors = []
for r in range(reps):
    preds = []
    for t in range(T):
        wvec = pj.abtovec(M_snaps[r][0][-1], M_snaps[r][1][-1])
        
        units = np.random.choice(veclen, pnum)
        # print(units)
        wvec[units] = np.float32(np.zeros(pnum))
        
        # ran = np.random.choice([0,1], veclen, p = [p, 1-p])
        # for i in range(veclen):
        #     wvec[i] = wvec[i] if ran[i] == 1 else np.float32(0)
        
        weights, biases = pj.cvectodict(wvec, pj.wkeys, pj.bkeys, pj.sw, pj.lw, pj.sb, pj.lb)
        
        # print(sum(sum(list(M_snaps[r][0][-1].values())[0] - list(weights.values())[0])))
        # print(sum(list(M_snaps[r][1][-1].values())[0] - list(biases.values())[0]))
        
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:        
            model = DNN.standard(DNN_dict, sess, seed = 1)
            
            pred = model.pred_w(x_scal, weights, biases)
            pred = data.scaler_y.inverse_transform(pred)
            
            preds.append(pred)
        
    DR_preds.append(preds)
    DR_preds_mean.append(np.mean(preds, 0))
    DR_preds_std.append(np.std(preds, 0).flatten())
    
    DR_errors.append(data.assess_pred(np.mean(preds, 0))[1])
    
print(len(M_snaps[0][0][-1]))

print(sum(sum(M_preds[0][-1] - DR_preds[0][1])))

#%%
import pickle 

with open('mcdrop_001_out.txt', 'wb') as f:
    pickle.dump([DR_preds, DR_preds_mean, DR_preds_std, DR_errors], f)  

# with open('mcdrop_01_out.txt', 'rb') as f:
#     [DR_preds, DR_preds_mean, DR_preds_std, DR_errors] = pickle.load(f)        
#%%
r = 1

ylim = [0, 2.2]

pred = DR_preds_mean[r]
error = data.assess_pred(pred)[-1]
print(sum(error))

plt.plot(x, error, '-', label = 'MC dropout (0.001)')
plt.xlabel('x', fontsize = 15)
plt.ylabel('Point-wise error', fontsize = 15)

plt.ylim(ylim)

plt.tight_layout()
plt.savefig('mcdrop_001_err.png', dpi = 300)
plt.show()

pred = pred.flatten()
pred_std = DR_preds_std[r]
plt.plot(x, pred, '-', label = 'MC dropout (0.001)')
plt.fill_between(x, pred-2*pred_std, pred+2*pred_std)
data.plot_eval_data(0)
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig('mcdrop_001_rep_fun.png', dpi = 300)
plt.show()
#%%
# pred = M_preds[r][-1]
# error = data.assess_pred(pred)[-1]
# print(sum(error))

# plt.plot(x, error, '-', label = 'single model')
# plt.xlabel('x', fontsize = 15)
# plt.ylabel('Point-wise error', fontsize = 15)

# plt.ylim(ylim)

# plt.tight_layout()
# plt.savefig('sm_rep_err.png', dpi = 300)
# plt.show()

# plt.plot(x, pred, '-', label = 'single model')
# data.plot_eval_data(0)
# plt.xlabel('x', fontsize = 15)
# plt.ylabel('y', fontsize = 15)
# plt.legend()
# plt.tight_layout()
# plt.savefig('sm_rep_fun.png', dpi = 300)
# plt.show()



