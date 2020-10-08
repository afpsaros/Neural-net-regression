# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:07:18 2020

@author: afpsaros
"""


class Callback:
    def __init__(self):
        self.model = None
        
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """Init after setting a model."""
            
    def on_train_begin(self):
        """Called at the beginning of training."""

    def at_step(self):
        """Called at step during training."""
 
    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""
       
    def on_epoch_end(self):
        """Called at the end of every epoch."""
        
    def on_train_end(self):
        """Called at the end of training."""

class CallbackList(Callback):

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.model = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def at_step(self):
        for callback in self.callbacks:
            callback.at_step()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def append(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)
        
class Snapper(Callback):

    def __init__(self, snap_step):
        super().__init__()
        self.snap_step = snap_step
    
    def on_train_begin(self):
            self.snap_weights = []
            self.snap_biases = []
            
            self.snap_epochs = []
    
    def at_step(self):
        if (self.model.epoch + 1) % self.snap_step == 0:    
            
            _snap_w, _snap_b = self.model.sess.run([self.model.weights, self.model.biases])
            
            self.snap_weights.append(_snap_w)
            self.snap_biases.append(_snap_b)
            
            self.snap_epochs.append(self.model.epoch)
        
    def get_snaps(self):
        return self.snap_weights, self.snap_biases

    def get_snap_epochs(self):
        return self.snap_epochs
    
class Losshistory(Callback):
    
    def on_train_begin(self):
        self.tr_error = []
        self.val_error = []
        
    def on_epoch_begin(self):
            self.val_error.append(self.model.sess.run(self.model.error_mean, \
                  feed_dict={self.model.X: self.model.Xv, self.model.Y: self.model.Yv}))
            self.tr_error.append(
            self.model.sess.run(self.model.error_mean, \
                                                           feed_dict={self.model.X: self.model.Xt, self.model.Y: self.model.Yt}))
        
    def on_train_end(self):
        self.on_epoch_begin()
               
    def get_loss_history(self):
        return self.tr_error, self.val_error
    
class LRschedule(Callback):
    
        def on_train_begin(self):
            self.lr_sched = []
        
        def on_epoch_begin(self):
            self.lr_sched.append(self.model.sess.run(self.model.optimizer._lr))
                    
        def get_lr_schedule(self):
            return self.lr_sched
    
    
    
    
    