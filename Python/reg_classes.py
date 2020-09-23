# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(afpsaros)s
"""
import tensorflow as tf
import numpy as np
import time

class scores:
    def errors(self):     
        error = tf.reduce_mean(tf.square(self.fp - self.Y))
        error_p = tf.math.abs(self.fp - self.Y)
        return error, error_p
                    
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
        
    
    def fit_from_dict(self, fit_dict):
        return self.fit(*list(fit_dict.values()))  

    def adv_fit_from_dict(self, fit_dict):
        return self.adv_fit(*list(fit_dict.values()))     
    
    def fit(self, initialize, wd_par, num_epochs, Xt, Yt, \
            Xv = None, Yv = None, lr = None, batches = None):
# =============================================================================
#     def fit(self, wd_par, num_epochs, Xt, Yt, \
#             Xv = None, Yv = None, lr = None, batches = None):
# =============================================================================
        self.initialize = initialize
        self.wd_par = wd_par
        self.num_epochs = num_epochs
        if self.initialize == 1:
# =============================================================================
#             self.xmin = min(Xt)
#             self.xmax = max(Yt)
# =============================================================================
            self.Xt = Xt
            self.Yt = Yt    
        self.lr = lr
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
         
        self.train_op = self.optimizer.minimize(self.obj)
               
        if self.initialize == 1:
            self.init = tf.global_variables_initializer()
            self.sess.run(tf.global_variables_initializer())
        else: 
            self.sess.run([self.init, tf.variables_initializer(self.optimizer.variables())])   
       
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
        for epoch in range(self.num_epochs): 
            self.sess.run(self.train_op, feed_dict={self.X: self.Xt, self.Y: self.Yt}) 

        return self
    
    def adv_fit(self, initialize, wd_par, num_epochs, Xt, Yt, Xv, Yv, lr = None):
        
        self.initialize = initialize
        self.wd_par = wd_par   
        self.num_epochs = num_epochs
        if self.initialize == 1:
# =============================================================================
#             self.xmin = min(Xt)
#             self.xmax = max(Yt)
# =============================================================================
            self.Xt = Xt
            self.Yt = Yt    
        self.Xv = Xv
        self.Yv = Yv
        self.lr = lr
        
        if self.initialize == 1:
            self.hyper_initial()
            self.fp = self.ffn()  
            self.error_mean = self.errors()[0]
        
        self.obj = self.error_mean + \
                        self.wd_par * tf.reduce_sum([tf.reduce_sum(tf.square(self.weights[i])) for i in self.weights])
                
        if self.lr is not None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_op = self.optimizer.minimize(self.obj)
        
        if self.initialize == 1:
            self.init = tf.global_variables_initializer()
            self.sess.run(tf.global_variables_initializer())
        else: 
            self.sess.run([self.init, tf.variables_initializer(self.optimizer.variables())])   
        
        self.check_w, self.check_b, self.tr_error, self.val_error = \
                    [[] for i in range(4)]   # not 4 * [[]] 
        
        for epoch in range(self.num_epochs):
            self.val_error.append(self.sess.run(self.error_mean, \
                  feed_dict={self.X: self.Xv, self.Y: self.Yv}))
            check_w_, check_b_, tr_error_, _ = \
            self.sess.run([self.weights, self.biases, self.error_mean, self.train_op], \
                                                           feed_dict={self.X: self.Xt, self.Y: self.Yt})
            
            self.check_w.append(check_w_)
            self.check_b.append(check_b_)
            self.tr_error.append(tr_error_)
           
        self.val_error.append(self.sess.run(self.error_mean, \
              feed_dict={self.X: self.Xv, self.Y: self.Yv}))
        check_w_, check_b_, tr_error_ = \
            self.sess.run([self.weights, self.biases, self.error_mean], \
                                                       feed_dict={self.X: self.Xt, self.Y: self.Yt})        
        self.check_w.append(check_w_)
        self.check_b.append(check_b_)
        self.tr_error.append(tr_error_)
               
        return self
    
# =============================================================================
#     def next_batch(self, b):
#         Xb = self.Xt[b * self.batch_size : (b + 1) * self.batch_size - 1]
#         Yb = self.Yt[b * self.batch_size : (b + 1) * self.batch_size - 1]
#         return Xb, Yb
# =============================================================================
    
class DNN(regressor):
    def __init__(self, widths, sess):
        
        super().__init__(sess)
        
        self.widths = widths
        self.layers = len(self.widths) - 2
        
        self.X = tf.placeholder(tf.float32, shape = [None, self.widths[0]])
        self.Y = tf.placeholder(tf.float32, shape = [None, self.widths[-1]])
        
        self.w_sizes = [[self.widths[i], self.widths[i + 1]] for i in range(len(self.widths) - 1)]
        self.w_keys = [('').join(('h', str(i + 1))) for i in range(self.layers)] + ['out']
        self.b_sizes = self.widths[1:]
        self.b_keys = [('').join(('b', str(i + 1))) for i in range(self.layers)] + ['out']
        
    def initialize(self, Xt, Yt):
# =============================================================================
#         self.xmin = min(Xt)
#         self.xmax = max(Yt)
# =============================================================================
        self.Xt = Xt
        self.Yt = Yt
        
        self.hyper_initial()
        self.fp = self.ffn()
        self.error_mean = self.errors()[0]
        self.init = tf.global_variables_initializer()
        
    
    def hyper_initial(self): 
        w_vars = []
        for s in self.w_sizes:
            std = np.sqrt(2 / (s[0] + s[1]))
            w_vars.append(tf.Variable(tf.random_normal(s, stddev = std)))        
        self.weights = dict(zip(self.w_keys, w_vars))

        b_vars = [tf.Variable(tf.zeros([s])) for s in self.b_sizes]
        self.biases = dict(zip(self.b_keys, b_vars))    

    def ffn(self):
        
# =============================================================================
#         self.A = 2.0 * (self.X - self.xmin) / (self.xmax - self.xmin) - 1.0
#         layer = tf.tanh(tf.add(tf.matmul(self.A, self.weights['h1']), self.biases['b1']))    
# =============================================================================
        layer = tf.tanh(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])) 
        for i in range(1, self.layers):
            layer = tf.tanh(tf.add(tf.matmul(layer, self.weights[self.w_keys[i]]), self.biases[self.b_keys[i]]))        
        layer = tf.add(tf.matmul(layer, self.weights['out']), self.biases['out'])
        return layer
    
    @classmethod
    def standard(cls, DNN_dict, sess):
        n_in, n_out, layers, width = list(DNN_dict.values())
        widths = [n_in] + layers * [width] + [n_out]
        
        return cls(widths, sess)
        