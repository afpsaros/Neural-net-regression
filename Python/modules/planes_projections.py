# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:36:59 2020

@author: afpsaros
"""

import tensorflow as tf
from reg_classes import DNN
import itertools
import numpy as np
flatten = itertools.chain.from_iterable

class planes_projections:
    
    def __init__(self, ws, bs):
        self.sw, self.lw = self.shapeslengths(ws[0])
        self.sb, self.lb = self.shapeslengths(bs[0])
        self.wkeys = ws[0].keys()
        self.bkeys = bs[0].keys()
    
    def dictovec(self, a):   
        # receives a dictionary and converts its values into a vector
        # val.flatten() flattens the matrix of weights of each layer
        # flatten from itertools joins the list of lists 
        # a: weight dictionary
        
        avec = np.array(list(flatten([val.flatten() for val in a.values()])))
        return avec
    
    def shapeslengths(self, a):
        # receives a dictionary and extracts the shape and the total number of weights in each layer
        # a: weight dictionary
        
        return [val.shape for val in a.values()], [len(val.flatten()) for val in a.values()]
    
    def vectodict(self, avec, keys, shapes, lengths):
        # receives a weight vector, the keys and the shapes for each layer
        # and outputs a weight dictionary
        # avec: vector of all the NN weights
        # keys: keys needed for dictionary - as obtained from .keys()
        # shapes: shape for each layer weight matrix
        # lengths: number of weights in each layer
        
        l = [] # creates a list of weight matrices
        k = 0
        for i, s in enumerate(shapes):
            l.append(avec[k : k + lengths[i]].reshape(s))
            k += lengths[i]
        a = dict(zip(keys, l))
        return a
    
    def abtovec(self, a, b):
        # receives weight and bias dictionaries and converts them into a weight+bias vector
        
        return np.concatenate((self.dictovec(a), self.dictovec(b)))
    
    def cvectodict(self, cvec, akeys, bkeys, sa, la, sb, lb):
        # receives a weight & bias vector as well as keys and shapes of weight and bias dictionaries
        # outputs weight & bias dictionaries
        # keys as obtained from .keys()
        
        avec = cvec[:sum(la)]
        bvec = cvec[sum(la):]
        
        return self.vectodict(avec, list(akeys), sa, la), self.vectodict(bvec, list(bkeys), sb, lb)
    
    def abcvectobasis(self, avec, bvec, cvec, wkeys, bkeys, sw, lw, sb, lb):

        # receives 3 weight & bias vectors as well as keys and shapes of weight and bias dictionaries
        # outputs basis vectors (weight+bias dictionaries)
    
        uvec = bvec - avec
        u_sq = sum([x**2 for x in uvec])
        u_norm = np.sqrt(u_sq)
        u_hat_vec = uvec / u_norm
        u_hat_w, u_hat_b = self.cvectodict(u_hat_vec, wkeys, bkeys, sw, lw, sb, lb)
    
        cmavec = cvec - avec
    
        inner = sum([x * y for x, y in zip(uvec, cmavec)])
    
        vvec = cmavec - inner / u_sq * uvec
        v_sq = sum([x**2 for x in vvec])
        v_norm = np.sqrt(v_sq)
        
        # print(u_hat_b_norm)
    
        v_hat_vec = vvec / v_norm
        v_hat_w, v_hat_b = self.cvectodict(v_hat_vec, wkeys, bkeys, sw, lw, sb, lb)
        
        return u_hat_vec, v_hat_vec, u_norm, v_norm, inner

    def projtoplane(self, wvec, wb1vec, u_hat_vec, v_hat_vec):
        # receives a weight & bias vector and the plane shift and basis vectors
        # outputs the x, y values corresponding to the projection of the vector onto the plane
        
        basis = np.concatenate((u_hat_vec.reshape(-1,1), v_hat_vec.reshape(-1,1)), axis = 1)
        ata = np.linalg.inv(np.matmul(basis.transpose(), basis))   
        prx, pry = np.matmul(ata, np.matmul(basis.transpose(), wvec - wb1vec))
        
        return prx, pry
    
    def createplane(self, ws, bs, pars, DNN_dict, tr_dict):
             
        if len(ws[0]) == 1:
            w1vec = np.float32(np.zeros(self.abtovec(ws[1], bs[1]).shape))
        else:
            w1vec = self.abtovec(ws[0], bs[0])
        w2vec = self.abtovec(ws[1], bs[1])
        w3vec = self.abtovec(ws[2], bs[2])
        
        u_hat_vec, v_hat_vec, u_norm, v_norm, inner = \
        self.abcvectobasis(w1vec, w2vec, w3vec, self.wkeys, self.bkeys, self.sw, self.lw, self.sb, self.lb)
                
        error_mat = []
        
        for par_1 in pars:
            error_v = []
            for par_2 in pars:        
                wvec_new = w1vec + par_1 * u_hat_vec + par_2 * v_hat_vec
                weights, biases = self.cvectodict(wvec_new, self.wkeys, self.bkeys, self.sw, self.lw, self.sb, self.lb)      
                
                g = tf.Graph()
                sess = tf.Session(graph = g)
                with g.as_default() as g:  
                    model = DNN.standard(DNN_dict, sess, seed = 1)
                    error_v.append(model.score_w(tr_dict, weights, biases)[0])
    
            error_mat.append(error_v)       
        
        _for_projection = (w1vec, u_hat_vec, v_hat_vec)
        _for_plot = (u_norm, v_norm, inner)
        
        return error_mat, _for_projection, _for_plot
    
    def projmultoplane(self, ws, bs, _for_projection):
        
        proj_x = []
        proj_y = []
        for c in range(len(ws)):
            _prx, _pry = self.projtoplane(self.abtovec(ws[c], bs[c]), *_for_projection)
            proj_x.append(_prx)
            proj_y.append(_pry)   
            
        return (proj_x, proj_y)
    
    def lineloss(self, ws, bs, pars, DNN_dict, tr_dict, inter, x):
      
        error_line = []
        w1vec = self.abtovec(ws[0], bs[0])
        w2vec = self.abtovec(ws[1], bs[1])
        for par in pars:
            wvec_new = par * w1vec + (1 - par) * w2vec
            weights, biases = self.cvectodict(wvec_new, self.wkeys, self.bkeys, self.sw, self.lw, self.sb, self.lb)
    
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:  
                model = DNN.standard(DNN_dict, sess, seed = 1)
                error_line.append(model.score_w(tr_dict, weights, biases)[0])
                if par == inter and inter is not None:
                    inter_pred = model.pred_w(x, weights, biases)
                else:
                    inter_pred = None
                                   
        return error_line, inter_pred
        
#%%
if __name__ == '__main__':
    
    import tensorflow as tf
    import matplotlib.pyplot as plt 
    from data_file import data_getter
    from reg_classes import DNN
    from callbacks import *
    
    n = 30
    s = 0
    val_split = 0.8
    n_eval = 100
    
    scale = 1
    
    data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
    data.plot_tr_data()
    data.plot_eval_data(1)
    

    DNN_dict = {
        'input dimension': 1,
        'output dimension': 1,
        'number of layers': 2,
        'layer width': 50, 
    }
    
    # fit_dict = {
    #     'initialize': 0,
    #     'wd_par': 0,
    #     'num_epochs': 5000,
    #     'Xt': data.Xt_scal,
    #     'Yt': data.Yt_scal,
    #     'Xv': data.Xv_scal,
    #     'Yv': data.Yv_scal,
    #     'lr': 0.01
    # }

    # eval_dict = {
    #     'Xe': data.Xe_scal,
    #     'Ye': data.Ye_scal
    # }  
        
    callbacks = []
    
    initial, final = 1, 1
    inifin = None if initial == 0 and final == 0 else InitialFinal(initial, final)
    if inifin is not None: callbacks.append(inifin)
    
    snap_step = None
    snap = None if snap_step is None else Snapper(snap_step)   
    if snap is not None: callbacks.append(snap) 
    
    loss_hist = 0
    loss = None if loss_hist == 0 else Losshistory()   
    if loss is not None: callbacks.append(loss) 
    
    fit_dict = {
        'callbacks': callbacks,
        'initialize': 1,
        'wd_par': 0,
        'num_epochs': 500,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
    }
#%%    
    opt_weights = []
    opt_biases = []
    init_weights = []
    init_biases = []
    
    num = 3
    
    for _ in range(num):
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:
            model = DNN.standard(DNN_dict, sess, seed = 0)

            model.fit_from_dict(fit_dict)    
            _iw, _ib, _fw, _fb = inifin.get_params()
            
            init_weights.append(_iw)
            init_biases.append(_ib)            
            opt_weights.append(_fw)
            opt_biases.append(_fb)    
#%%
    pars = np.linspace(-5, 22, 10)
    
    tr_dict = {
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal
    }  
    
    pj = planes_projections(opt_weights, opt_biases)
    
    error_mat, _for_projection, (u_norm, v_norm, inner) = \
        pj.createplane(opt_weights, opt_biases, pars, DNN_dict, tr_dict)
            
    # sw, lw = pj.shapeslengths(opt_weights[0])
    # sb, lb = pj.shapeslengths(opt_biases[0])
    # wkeys = opt_weights[0].keys()
    # bkeys = opt_biases[0].keys()
    
    # w1vec = pj.abtovec(opt_weights[0], opt_biases[0])
    # w2vec = pj.abtovec(opt_weights[1], opt_biases[1])
    # w3vec = pj.abtovec(opt_weights[2], opt_biases[2])
    
    # u_hat_vec, v_hat_vec, u_norm, v_norm, inner = \
    # pj.abcvectobasis(w1vec, w2vec, w3vec, wkeys, bkeys, sw, lw, sb, lb)
    

    # for par_1 in pars:
    #     error_v = []
    #     for par_2 in pars:        
    #         wvec_new = w1vec + par_1 * u_hat_vec + par_2 * v_hat_vec
    #         weights, biases = pj.cvectodict(wvec_new, wkeys, bkeys, sw, lw, sb, lb)      
            
    #         g = tf.Graph()
    #         sess = tf.Session(graph = g)
    #         with g.as_default() as g:  
    #             model = DNN.standard(DNN_dict, sess, seed = 1)
    #             error_v.append(model.score_w(tr_dict, weights, biases)[0])

    #     error_mat.append(error_v)             
#%%
    # basis = np.concatenate((u_hat_vec.reshape(-1,1), v_hat_vec.reshape(-1,1)), axis = 1)
    # ata = np.linalg.inv(np.matmul(basis.transpose(), basis))
    
    # init_proj_x = []
    # init_proj_y = []
    
    # for c in range(num):
    #     # prx, pry = pj.projtoplane(pj.abtovec(init_weights[c], init_biases[c]), w1vec, u_hat_vec, v_hat_vec)
    #     prx, pry = pj.projtoplane(pj.abtovec(init_weights[c], init_biases[c]), *_for_projection)
    #     init_proj_x.append(prx)
    #     init_proj_y.append(pry)  
        
    init_proj = pj.projmultoplane(init_weights, init_biases, _for_projection)
    
    xx, yy = np.meshgrid(pars, pars)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training loss contour plots', fontsize = 30)
    
    im1 = ax1.contourf(xx, yy, np.array(error_mat).transpose(), 200, origin='lower', cmap='RdGy')
    fig.colorbar(im1, ax=ax1)
    ax1.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], marker = 'x', color = 'k', s = 50, label = 'final')
    # ax1.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
    ax1.scatter(*init_proj, marker = 'x', color = 'm', s = 50, label = 'initial')
    ax1.legend();
    
    im2 = ax2.contour(xx, yy, np.array(error_mat).transpose(), 40, origin='lower', cmap='RdGy')
    fig.colorbar(im2, ax=ax2)
    ax2.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], color = 'k', marker = 'x', s = 50, label = 'final')
    # ax2.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
    ax2.scatter(*init_proj, marker = 'x', color = 'm', s = 50, label = 'initial')
    ax2.legend();        
    
#%%
    combs = [[0, 1], [1, 2], [0, 2]]
    pars = np.linspace(0, 1, 11)
    
    inters = []
    x = data.Xe.reshape(-1,1)
    x_scal = data.Xe_scal
    
    inter = 0.5
    plt.title('Loss on the line connecting 2 optima', fontsize = 20)
    for c in range(3):
        
        _ws = [opt_weights[combs[c][0]], opt_weights[combs[c][1]]]
        _bs = [opt_biases[combs[c][0]], opt_biases[combs[c][1]]]
        
        error_line, inter_pred = pj.lineloss(_ws, _bs, pars, DNN_dict, tr_dict, inter, x_scal)
        inter_pred = data.scaler_y.inverse_transform(inter_pred)
        inters.append(inter_pred)
        
        # error_line = []
        # w1vec = pj.abtovec(opt_weights[combs[c][0]], opt_biases[combs[c][0]])
        # w2vec = pj.abtovec(opt_weights[combs[c][1]], opt_biases[combs[c][1]])
        # for par in pars:
        #     wvec_new = par * w1vec + (1 - par) * w2vec
        #     weights, biases = pj.cvectodict(wvec_new, wkeys, bkeys, sw, lw, sb, lb)
    
        #     g = tf.Graph()
        #     sess = tf.Session(graph = g)
        #     with g.as_default() as g:  
        #         model = DNN.standard(DNN_dict, sess, seed = 1)
        #         error_line.append(model.score_w(tr_dict, weights, biases)[0])
                
        #         if par == .5:
        #             pred = model.pred_w(x_scal, weights, biases)
        #             pred = data.scaler_y.inverse_transform(pred)
        #             inter_pred.append(pred)
   
        plt.plot(pars, error_line, '-o', label = 'optima %.1d and %.1d' %(combs[c][0]+1, combs[c][1]+1))
    
    plt.legend(bbox_to_anchor=(1.4, 1.0))
    plt.show()   
    
    plt.title('Intermediate predictions', fontsize = 20)
    for c in range(3):
        plt.plot(x, inters[c], label = 'combo {}'.format(combs[c]))
        
    plt.legend(bbox_to_anchor=(1.4, 1.0))
    plt.show()   
#%%   
    plt.title('Predictions', fontsize = 20)
    # plt.axvline(x = 0, label = 'data bounds', color = 'r')
    # plt.axvline(x = 10, color = 'r')
    for c in range(len(opt_weights)):
        weights = opt_weights[c]
        biases = opt_biases[c]
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:  
            model = DNN.standard(DNN_dict, sess, seed = 1)
            error_line.append(model.score_w(tr_dict, weights, biases)[0])
            pred = model.pred_w(x_scal, weights, biases)
            pred = data.scaler_y.inverse_transform(pred)
            
        plt.plot(x, pred, label = 'optimum %.1d' %(c+1))
        
    data.plot_eval_data(0)  
    plt.legend(bbox_to_anchor=(1.4, 1.0))
    plt.show()    
#%%
    plt.title('Test error for optima and ensemble', fontsize = 20)
    plt.yscale('log')
    # plt.axvline(x = 0, label = 'data bounds', color = 'r')
    # plt.axvline(x = 10, color = 'r')
    ensemble = np.zeros(x.shape)
    for c in range(len(opt_weights)):
        weights = opt_weights[c]
        biases = opt_biases[c]
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:  
            model = DNN.standard(DNN_dict, sess, seed = 1)
            
            pred = model.pred_w(x_scal, weights, biases)
            pred = data.scaler_y.inverse_transform(pred)
            
            temp_error = data.assess_pred(pred)[-1]
            print('error {}'.format(c+1), data.assess_pred(pred)[0])
    
        ensemble += pred / num
        plt.plot(x, temp_error, label = 'optimum %.1d' %(c+1))
    
    ens_error = data.assess_pred(ensemble)[-1]
    print('ensemble error', data.assess_pred(ensemble)[0])
    plt.plot(x, ens_error, label = 'ensemble')
    plt.legend(bbox_to_anchor=(1.4, 1.0))
    plt.show()
#%%   
    callbacks = []
    
    initial, final = 1, 1
    inifin = None if initial == 0 and final == 0 else InitialFinal(initial, final)
    if inifin is not None: callbacks.append(inifin)
    
    snap_step = 50
    snap = None if snap_step is None else Snapper(snap_step)   
    if snap is not None: callbacks.append(snap) 
    
    loss_hist = 0
    loss = None if loss_hist == 0 else Losshistory()   
    if loss is not None: callbacks.append(loss) 
    
    fit_dict = {
        'callbacks': callbacks,
        'initialize': 1,
        'wd_par': 0,
        'num_epochs': 500,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
    }
    
    snaps = list(range(num))
    
    snap_weights = {k: [] for k in snaps} # be careful! not zip!
    snap_biases = {k: [] for k in snaps}
    opt_weights = []
    opt_biases = []
    init_weights = []
    init_biases = []
    
    num = 3
    
    for c in range(num):
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:
            model = DNN.standard(DNN_dict, sess, seed = 0)
            
            model.fit_from_dict(fit_dict)    
            
            _iw, _ib, _fw, _fb = inifin.get_params()
            
            init_weights.append(_iw)
            init_biases.append(_ib)            
            opt_weights.append(_fw)
            opt_biases.append(_fb)    
            
            _sw, _sb = snap.get_snaps()
            
            snap_weights[c] = _sw
            snap_biases[c] = _sb

            
#%%           
# =============================================================================
#     sw, lw = pj.shapeslengths(opt_weights[0])
#     sb, lb = pj.shapeslengths(opt_biases[0])
#     wkeys = opt_weights[0].keys()
#     bkeys = opt_biases[0].keys()
#     
#     w1vec = pj.abtovec(opt_weights[0], opt_biases[0])
#     w2vec = pj.abtovec(opt_weights[1], opt_biases[1])
#     w3vec = pj.abtovec(opt_weights[2], opt_biases[2])
# =============================================================================
#%% 
    pj = planes_projections()
    zerovec = np.float32(np.zeros(pj.abtovec(opt_weights[0], opt_biases[0]).shape))
    
    w1vec = zerovec
    # zw, zb = pj.cvectodict(zerovec, wkeys, bkeys, sw, lw, sb, lb)
    # zwbvec = pj.abtovec(zw, zb)
    
    w2vec = pj.abtovec(opt_weights[1], opt_biases[1])
    w3vec = pj.abtovec(opt_weights[2], opt_biases[2])
        
    u_hat_vec, v_hat_vec, u_norm, v_norm, inner = \
    pj.abcvectobasis(w1vec, w2vec, w3vec, wkeys, bkeys, sw, lw, sb, lb)
#%%    
    pars = np.linspace(-2, 15, 30)
    error_mat = []
    
    tr_dict = {
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal
    }  
    for par_1 in pars:
        error_v = []
        for par_2 in pars:        
            wvec_new = w1vec + par_1 * u_hat_vec + par_2 * v_hat_vec
            weights, biases = pj.cvectodict(wvec_new, wkeys, bkeys, sw, lw, sb, lb)      
            
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:  
                model = DNN.standard(DNN_dict, sess, seed = 1)
                error_v.append(model.score_w(tr_dict, weights, biases)[0])
                
                
    
        error_mat.append(error_v)             
#%%
    basis = np.concatenate((u_hat_vec.reshape(-1,1), v_hat_vec.reshape(-1,1)), axis = 1)
    ata = np.linalg.inv(np.matmul(basis.transpose(), basis))
    
    snap_proj_x = []
    snap_proj_y = []
    init_proj_x = []
    init_proj_y = []
    
    for c in [1, 2]:
        prx, pry = pj.projtoplane(pj.abtovec(init_weights[c], init_biases[c]), w1vec, u_hat_vec, v_hat_vec)
        init_proj_x.append(prx)
        init_proj_y.append(pry)    
        for snap in range(len(snap_weights[0])): 
            prx, pry = pj.projtoplane(pj.abtovec(snap_weights[c][snap], snap_biases[c][snap]), w1vec, u_hat_vec, v_hat_vec)
            snap_proj_x.append(prx)
            snap_proj_y.append(pry)
    
    xx, yy = np.meshgrid(pars, pars)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training loss contour plots', fontsize = 30)
    
    im1 = ax1.contourf(xx, yy, np.array(error_mat).transpose(), 200, origin='lower', cmap='RdGy')
    fig.colorbar(im1, ax=ax1)
    ax1.scatter([0], [0], marker = 's', s = 50, label = 'origin')
    ax1.scatter(init_proj_x, init_proj_y, marker = 'x', s = 50, label = 'initial')
    ax1.scatter(snap_proj_x, snap_proj_y, marker = '.', s = 15, label = 'snaps')
    ax1.scatter([u_norm, inner / u_norm], [0, v_norm], marker = 'x', color = 'k', s = 50, label = 'final')
    ax1.legend();
    
    im2 = ax2.contour(xx, yy, np.array(error_mat).transpose(), 40, origin='lower', cmap='RdGy')
    fig.colorbar(im2, ax=ax2)
    ax2.scatter([0], [0], marker = 's', s = 50, label = 'origin')
    ax2.scatter(init_proj_x, init_proj_y, marker = 'x', s = 50, label = 'initial')
    ax2.scatter(snap_proj_x, snap_proj_y, marker = '.', s = 15, label = 'snaps')
    ax2.scatter([u_norm, inner / u_norm], [0, v_norm], color = 'k', marker = 'x', s = 50, label = 'final')
    ax2.legend();     
            