# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:47:57 2020

@author: afpsaros
"""



from collections import defaultdict
import numpy as np
from reg_classes import DNN
import tensorflow as tf
import time
from callbacks import *

class random_global_cv:
    
    def __init__(self, ev_params, ev_arch, refit, r_callbacks, n_random, random_seed):
        self.ev_params = ev_params
        self.ev_arch = ev_arch
        
        self.refit = refit
        self.r_callbacks = r_callbacks
        
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
            
            # print(l1)
            
            l2 = []
            
            self.snap_step = None
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
                    
                    if key == 'callbacks':
                        self.snap_step = ev_fit_dict[key]
                        snap = Snapper(self.snap_step)   
                        callbacks = [snap]
                        ev_fit_dict[key] = callbacks                          
                    
                elif val[1] == 'c':
                    ev_fit_dict[key] = int(10 ** np.random.uniform(val[0][0], val[0][1]))
                    l2.append(ev_fit_dict[key])    
                elif val[1] == 'd':
                    
                    ev_fit_dict[key] = 10 ** np.random.uniform(val[0][0], val[0][1])
                    l2.append(ev_fit_dict[key]) 
                    
                    if key == 'decay':
                        ev_fit_dict[key] = ['cosine_restarts',self.snap_step, ev_fit_dict[key], 1., 1.]
                        self.snap_step = None
                    
                elif val[1] == 'e':
                    
                    sel = np.random.choice([0, 1])
                    if sel == 0:
                        ev_fit_dict[key] = val[0][0]
                    else: 
                        ev_fit_dict[key] = 10 ** np.random.uniform(val[0][1], val[0][2])
                    l2.append(ev_fit_dict[key])     
                    
                # elif val[1] == 'f':            
                #     # print(val[0])
                #     sel_snap_step = np.random.choice(val[0])
                #     # print(sel_snap_step)
                    
                #     l2.append(sel_snap_step) 
                #     snap = Snapper(sel_snap_step)   
                #     callbacks = [snap]
                #     ev_fit_dict['callbacks'] = callbacks                   
                    
                # elif val[1] == 'g':  
                #     ratio = 10 ** np.random.uniform(val[0][0], val[0][1])
                #     l2.append(ratio)
                #     ev_fit_dict[key] = ['cosine_restarts',sel_snap_step, ratio, 1., 1.]
                #     sel_snap_step = None
            
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
                if self.random_seed in [0, 1]:
                    model = DNN.standard(ev_DNN_dict, sess, self.random_seed)
                else:
                    sel = np.random.choice(np.arange(1, self.random_seed + 1)) 
                    model = DNN.standard(ev_DNN_dict, sess, sel)
                 
                model.fit_from_dict(ev_fit_dict)
                
                
                # self.scores[tuple([tuple(l1), tuple(l2)])] = \
                # model.fit_from_dict(ev_fit_dict).score(val_dict)[0] 

##############################################################################################################                
                snap_weights, snap_biases = snap.get_snaps()
                ensemble = model.fun_ensemble(snap_weights, snap_biases)
                self.scores[tuple([tuple(l1), tuple(l2)])] = model.score_ens(val_dict, ensemble)
##############################################################################################################  
                
            self.tocs.append(time.perf_counter()-tic)
            print(time.perf_counter()-tic)
            
        self.best = self.minscore()
                

        if self.refit == 1: 
            
            ev_fit_dict['callbacks'] = self.r_callbacks
                        
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:
            
                for i, key in enumerate(self.ev_arch.keys()):
                    ev_DNN_dict[key] = self.best[0][0][i]
                    # print('best arch', key, ev_DNN_dict[key])
                    
                for i, key in enumerate(self.ev_params.keys()):
                    ev_fit_dict[key] = self.best[0][1][i]       
                    # print('best params', key, ev_fit_dict[key])

                    if key == 'callbacks':
                        self.snap_step = ev_fit_dict[key]
                        snap = Snapper(self.snap_step)   
                        callbacks = [snap]
                        ev_fit_dict[key] = callbacks   
                        
                    elif key == 'decay':
                        ev_fit_dict[key] = ['cosine_restarts',self.snap_step, ev_fit_dict[key], 1., 1.]
                        self.snap_step = None                        
                    
                if self.random_seed in [0, 1]:
                    model = DNN.standard(ev_DNN_dict, sess, self.random_seed)
                else:
                    sel = np.random.choice(np.arange(1, self.random_seed + 1)) 
                    model = DNN.standard(ev_DNN_dict, sess, sel)
                
                model.fit_from_dict(ev_fit_dict)
                snap_weights, snap_biases = snap.get_snaps()
                model.fun_ensemble(snap_weights, snap_biases)
                    
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
    nu = 0.1 / np.pi
    scale = 1
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(scale)
    data.plot3D_train()
    data.plot3D_eval(1)
    
    data.plot2D_eval(1)
#%%    
    DNN_dict = {
        'input dimension': 2,
        'output dimension': 1,
        'number of layers': 4,
        'layer width': 20, 
    }

    # callbacks = []
    
    # snap_step = None
    # snap = None if snap_step is None else Snapper(snap_step)   
    # if snap is not None: callbacks.append(snap) 
    
    # loss_hist = 0
    # loss = None if loss_hist == 0 else Losshistory()   
    # if loss is not None: callbacks.append(loss) 
    
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

    refit = 1
    r_callbacks = None  
    n_random = 5
    random_seed = 3

    # ev_arch = {
    #         'number of layers': ([8], 'a'),
    #         'layer width': ([80], 'a')
    #         }
    ev_arch = {}
        # 'wd_par': ([0, -5, -3], 'e'),
    ev_params = {
        'num_epochs': ([10000], 'b'),
        'lr': ([-3, -2], 'd'),
        'callbacks': ([2000, 5000], 'b'),
        'decay': ([-2, -1], 'd')
        }                    
#%% 
    
    arch_cv = random_global_cv(ev_params, ev_arch, refit, r_callbacks, n_random, random_seed)
    scores, best, model = arch_cv.fit(fit_dict, DNN_dict)
    
    print(arch_cv.best)
    
#%%
    plt.plot(arch_cv.tocs) 
    plt.show()
    
    data.create_eval_data_3D(nt_eval = 5).preproc(scale)
    
    xlen = data.xs
    
    data.plot2D_eval(0)
    x = data.Xe[:xlen, 0]
    if scale== 1:
        for i in range(len(data.times)):
            
            pred = data.scaler_y.inverse_transform(model.pred_ens(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2], model.ensemble))
            plt.plot(x.reshape(-1,1), pred, '.')
            
    else:
        for i in range(len(data.times)):
            pred = model.pred(data.Xe_scal[i * xlen:xlen + i * xlen, 0:2], model.ensemble)
            plt.plot(x, pred, '.')
    
    plt.show()      
      
    pred = model.pred_ens(data.Xe_scal, model.ensemble)
    pred = data.scaler_y.inverse_transform(pred)
    print(data.assess_pred(pred)[0])    

    plt.plot(data.assess_pred(pred)[1])      
    print(np.mean(data.assess_pred(pred)[1]))
                
                
    print(max(data.assess_pred(pred)[1]))                
                
     