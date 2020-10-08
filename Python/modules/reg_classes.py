# -*- coding: utf-8 -*-
"""
Created on Sep 20 2020

@author: afpsaros
"""

import tensorflow as tf
import numpy as np
import time

from callbacks import *

class scores:
    def errors(self):     
        error = tf.reduce_mean(tf.square(self.fp - self.Y))
        error_p = tf.math.abs(self.fp - self.Y)
        return error, error_p

    def pred_ens(self, Xe, ens):
        return self.sess.run(ens, feed_dict={self.X: Xe})
                    
    def pred(self, Xe):
        return self.sess.run(self.fp, feed_dict={self.X: Xe})

    def pred_w(self, Xe, W, b):
        self.weights, self.biases = W, b
        self.fp = self.ffn()
        return self.sess.run(self.fp, feed_dict={self.X: Xe})
    
    def score(self, eval_dict):
        Xe, Ye = list(eval_dict.values())
        return self.sess.run(self.errors(), feed_dict = {self.X: Xe, self.Y: Ye})   
    
    def score_w(self, eval_dict, W, b):
        Xe, Ye = list(eval_dict.values())
        self.weights, self.biases = W, b
        self.fp = self.ffn()
        return self.sess.run(self.errors(), feed_dict = {self.X: Xe, self.Y: Ye})  
    
    
    
class regressor(scores):
    
    def __init__(self, sess):
        self.sess = sess
        self.optimizer = tf.train.AdamOptimizer()
        
    def close_sess(self):
        self.sess.close()

    def _get_learningrate(self, lr, decay):
        if decay is None:
            return lr, None
        global_step = tf.Variable(0, trainable=False)
        return (
            {
                "inverse time": tf.train.inverse_time_decay(
                    lr, global_step, decay[1], decay[2]
                ),
                "cosine": tf.train.cosine_decay(lr, global_step, decay[1], alpha=decay[2]),
                "cosine_restarts": tf.train.cosine_decay_restarts(
                    lr, global_step, decay[1], t_mul=decay[3], m_mul=decay[4], alpha=decay[2])
            }[decay[0]],
            global_step,
        )
    
    def fun_test(self, W, b):
        self.weights, self.biases = W, b
        self.fp = self.ffn()
        return self
        
    def fit_from_dict(self, fit_dict):
        return self.fit(*list(fit_dict.values()))  
      
    def fit(self, callbacks, initialize, wd_par, num_epochs, Xt, Yt, \
            Xv = None, Yv = None, lr = None, decay = None, batches = None):

        self.initialize = initialize
        self.wd_par = wd_par
        self.num_epochs = num_epochs
        if self.initialize == 1:
            self.Xt = Xt
            self.Yt = Yt    
        
        self.Xv = Xv
        self.Yv = Yv
        
        self.lr, self.global_step = self._get_learningrate(lr, decay)
##############################################        
        self.callbacks = CallbackList(callbacks = callbacks)
        self.callbacks.set_model(self)
# =============================================================================
#         self.batches = batches
#         
#         if self.batches is not None:
#             self.batch_size = self.Xt.shape[0] // self.batches
#             self.itpep = self.Xt.shape[0] // self.batch_size
# ============================================================================= 
        
        if self.initialize == 1:
            self.hyper_initial()
            self.fp = self.ffn()  
            self.error_mean = self.errors()[0]
        
        self.obj = self.error_mean + \
                        self.wd_par * tf.reduce_sum([tf.reduce_sum(tf.square(self.weights[i])) for i in self.weights])
        
        if self.lr is not None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
         
        self.train_op = self.optimizer.minimize(self.obj, global_step=self.global_step)
               
        if self.initialize == 1:
            self.init = tf.global_variables_initializer()
            self.sess.run(tf.global_variables_initializer())
        else: 
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))   
# =============================================================================
#             self.sess.run([self.init, tf.variables_initializer(self.optimizer.variables())])   
# =============================================================================


# =============================================================================
#         if self.batches is not None:
#             for epoch in range(self.num_epochs): 
#                 for it in range(self.itpep):
#                     Xb, Yb = self.next_batch(it)
#                     self.sess.run(self.train_op, feed_dict={self.X: Xb, self.Y: Yb}) 
#                     
#         else:
#             for epoch in range(self.num_epochs): 
#                 self.sess.run(self.train_op, feed_dict={self.X: self.Xt, self.Y: self.Yt}) 
# =============================================================================        
        self.callbacks.on_train_begin()
        
        for self.epoch in range(self.num_epochs): 
            # print(sess.run(self.optimizer._lr))
            self.callbacks.on_epoch_begin()
            
            self.sess.run(self.train_op, feed_dict={self.X: self.Xt, self.Y: self.Yt}) 
                      
            self.callbacks.at_step()
            
            self.callbacks.on_epoch_end()
                                
        self.callbacks.on_train_end()    
        
        return self
    
class DNN(regressor):
    def __init__(self, widths, sess, seed):
        
        super().__init__(sess)
        
        self.seed = seed
        self.widths = widths
        self.layers = len(self.widths) - 2
        
        self.X = tf.placeholder(tf.float32, shape = [None, self.widths[0]])
        self.Y = tf.placeholder(tf.float32, shape = [None, self.widths[-1]])
        
        self.w_sizes = [[self.widths[i], self.widths[i + 1]] for i in range(len(self.widths) - 1)]
        self.w_keys = [('').join(('h', str(i + 1))) for i in range(self.layers)] + ['out']
        self.b_sizes = self.widths[1:]
        self.b_keys = [('').join(('b', str(i + 1))) for i in range(self.layers)] + ['out']
        
    def initialize(self, Xt, Yt):

        self.Xt = Xt
        self.Yt = Yt
        
        self.hyper_initial()
        self.fp = self.ffn()
        self.error_mean = self.errors()[0]
        self.init = tf.global_variables_initializer()     
        
        self.sess.run(self.init)   
    
    def hyper_initial(self): 
        w_vars = []
        for s in self.w_sizes:
            std = np.sqrt(2 / (s[0] + s[1]))
            
            if self.seed == 0:
                w_vars.append(tf.Variable(tf.random_normal(s, stddev = std)))  
            else:
                w_vars.append(tf.Variable(tf.random_normal(s, stddev = std, seed = self.seed)))  
        self.weights = dict(zip(self.w_keys, w_vars))

        b_vars = [tf.Variable(tf.zeros([s])) for s in self.b_sizes]
        self.biases = dict(zip(self.b_keys, b_vars))    

    def ffn(self):
  
        layer = tf.tanh(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])) 
        for i in range(1, self.layers):
            layer = tf.tanh(tf.add(tf.matmul(layer, self.weights[self.w_keys[i]]), self.biases[self.b_keys[i]]))        
        layer = tf.add(tf.matmul(layer, self.weights['out']), self.biases['out'])
        return layer
    
    @classmethod
    def standard(cls, DNN_dict, sess, seed):
        n_in, n_out, layers, width = list(DNN_dict.values())
        widths = [n_in] + layers * [width] + [n_out]
        
        return cls(widths, sess, seed)

#%%
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    from data_file import data_getter
    
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

    callbacks = []
    
    snap_step = 100
    snap = None if snap_step is None else Snapper(snap_step)   
    if snap is not None: callbacks.append(snap) 
    
    loss_hist = 1
    loss = None if loss_hist == 0 else Losshistory()   
    if loss is not None: callbacks.append(loss) 

    fit_dict = {
        'callbacks': callbacks,
        'initialize': 1,
        'wd_par': 0,
        'num_epochs': 1000,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
        'Xv': data.Xv_scal,
        'Yv': data.Yv_scal,
        'lr': 0.01,
        'decay': None,
    }
    
    fit_dict['decay'] = ['cosine_restarts',snap_step, 0.002, 1., 1.]
    
    # eval_dict = {
    #     'Xe': data.Xe_scal,
    #     'Ye': data.Ye_scal
    # }  

    sess = tf.Session()
    model = DNN.standard(DNN_dict, sess, seed = 1)
    model.initialize(fit_dict['Xt'], fit_dict['Yt'])
    
    model.fit_from_dict(fit_dict)
        

    snap_weights, snap_biases = snap.get_snaps()
    tr_error, val_error = loss.get_loss_history()
    
    plt.yscale('log')
    plt.plot(tr_error, label = 'training loss')
    plt.plot(val_error, label = 'validation error')
    plt.legend()
    plt.show()
       
    if scale == 1:
        x = data.Xe.reshape(-1,1)
        x_scal = data.Xe_scal
        pred = model.pred(x_scal)
        pred = data.scaler_y.inverse_transform(pred)
        plt.plot(x, pred, '.')
        data.plot_eval_data(1)
        
        print(data.assess_pred(pred)[0])
        plt.plot(x, data.assess_pred(pred)[1])        

       
    else:
        x = data.Xe.reshape(-1,1)
        pred = model.pred(x)
        plt.plot(x, pred, '.')
        data.plot_eval_data(1)     
        
        print(data.assess_pred(pred)[0])
        plt.plot(x, data.assess_pred(pred)[1])
        plt.show()        

    for i in range(len(snap_weights)):
        pred = model.pred_w(x_scal, snap_weights[i], snap_biases[i])
        pred = data.scaler_y.inverse_transform(pred)
        plt.plot(x, pred, '.', label = '{}'.format(i))
        
    data.plot_eval_data(0)  
    plt.legend()    
    plt.show()
#%%    
    # tocs = []
    
    # tic1 = time.perf_counter()
    # for i in range(fit_dict['num_epochs']):
    #     tic = time.perf_counter()
        
    #     g = tf.Graph()
    #     sess = tf.Session(graph = g)
    #     with g.as_default() as g:
    #         model = DNN.standard(DNN_dict, sess, seed = 1)        
    #         pred = model.pred_w(x_scal, check_w[i], check_b[i])
    #     tocs.append(time.perf_counter() - tic)
     
    # print(time.perf_counter() - tic1)    
    # plt.plot(tocs)
    # plt.show()

    




        