#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Couches du réseau de neurones
# code : Lara Perinetti

from abc import ABC
import numpy as np
from collections import defaultdict



# ---------------------- Layers ---------------------

class Layer(ABC):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        #self.res_forward = None # result after activation
        #self.combi_lin = None # result before activation

    def forward_activation(self, x):
        self.combi_lin = np.add(np.matmul(x, self.W), self.b)
        self.res_forward = self.activation()
        return self.res_forward
    
    def backward_activation(self, backward, HiddenL):
        bgrad = backward.dot(self.gradient_activation())
        Wgrad = HiddenL.T.dot(bgrad)
        assert bgrad.shape == self.b.shape
        assert Wgrad.shape == self.W.shape
        return Wgrad, bgrad


class Linear_Layer(Layer):
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "Linear Layer"
    
    def activation(self):
        return self.combi_lin
    
    def gradient_activation(self):
        return np.identity(self.b.shape[1])


class Softmax_Layer(Layer):
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "SOFTMAX"

    def forward_activation(self, x):
        self.combi_lin = np.add(np.matmul(x, self.W), self.b)

    def activation(self):
        max_x = self.combi_lin - np.max(self.combi_lin) #with max, more stable
        exp = np.exp(max_x)
        return exp / np.sum(exp)
    
    def backward_activation(self, y, HiddenL):
        bgrad = self.activation() - y
        Wgrad = np.matmul(HiddenL.res_forward.T, bgrad)
        assert Wgrad.shape == self.W.shape
        assert bgrad.shape == self.b.shape
        return (Wgrad, bgrad)


class RelU_Layer(Layer):
    """Layer with RelU activation function"""
    def __init__(self, W, b):
        Linear_Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "RELU"
    
    def activation(self):
        return self.combi_lin * (self.combi_lin > 0.0)
    
    def gradient_activation(self):
        return np.diag((1.0 * (self.combi_lin > 0.0))[0])


class Tanh_Layer(Layer):
    """Layer with RelU activation function"""
    def __init__(self, W, b):
        Linear_Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "TanH"
    
    def activation(self):
        return (np.exp(self.combi_lin) - np.exp(-self.combi_lin)) / (np.exp(self.combi_lin) + np.exp(-self.combi_lin))
    
    def gradient_activation(self):
        return np.diag((1.0 - (self.combi_lin ** 2.0))[0])


