#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti

import numpy as np
import read_connl as rc
import math
import random
import argparse
from time import time
from collections import defaultdict

# ---------------------- Constants ---------------------
UNKNOWN= "UNKNOWN"

RELU = "relu"
SOFTMAX = "softmax"
TANH = "tanh"
LIN = "linear"

# ---------------------- Layers ---------------------

class Layer(): #abstract
    def __init__(self, W,b):
        self.W = W
        self.b = b
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation
    
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
    """ Layer with linear activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "Linear Layer "
    
    def forward_activation(self):
        return self.combi_lin
    
    def backward_activation(self):
        return np.identity(self.b.shape[1])


class Softmax_Layer(Layer):
    """ Layer with sigmoid activation function 
        Always the output layer """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return "Softmax Layer "
    
    def forward_activation(self, x):
        self.combi_lin = np.add(np.matmul(x,self.W), self.b)

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
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None # result after activation
        self.combi_lin = None # result before activation

    def __str__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
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


# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, hidden_layers, hidden_activations, classes, learning_rate, xavier=True): 
        """ Neural Network 
            @param hidden_layers            liste du nombre de neurones par layer
            @param hidden_activations       liste des fonctions d'activation pour chaque layer
            @param classes                  liste des classes
            @param learning_rate            default=0.01
            @param xavier                   initialiser les paramètres selon la méthode de xavier 
        """ 
        self.learning_rate = learning_rate
        self.T = 1 #optimization of the learning rate, set to 1

        self.hidden_number = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations
        self.classes = classes

        self.input_size = 2
        self.hidden = [self.init_linear_layer(self.input_size, self.hidden_layers[0], self.hidden_activations[0])]
        for i in range(self.hidden_number-1):
            self.hidden.append(self.init_linear_layer(self.hidden_layers[i], self.hidden_layers[i+1], self.hidden_activations[i+1]))
        self.output = self.init_linear_layer(hidden_layers[-1], len(classes), SOFTMAX)



    # ---------------------- Init parameters ---------------------

    def init_linear_layer(self, input_size, output_size, activ, xavier=True):
        bfr_b = np.zeros((1, output_size))
        if xavier:
            mini = -(math.sqrt(6) / (math.sqrt(input_size + output_size)))
            maxi = math.sqrt(6) / math.sqrt(input_size + output_size)
            bfr_W = np.random.uniform(mini, maxi, (input_size, output_size))
        else:
            bfr_W = np.random.normal(0, math.sqrt(2 / input_size), (input_size, output_size))
        
        if activ == SOFTMAX:
            return Softmax_Layer(bfr_W, bfr_b)
        elif activ == RELU:
            return RelU_Layer(bfr_W, bfr_b)
        elif activ == LIN:
            return Linear_Layer(bfr_W, bfr_b)
        elif activ == TANH:
            return Tanh_Layer(bfr_W, bfr_b)



    # ---------------------- Propagations ---------------------

    def forward_propagation(self, x):
        """
        Prediction of the NN.
        Input data, X,  is “forward propagated” through the network 
            layer by layer to the final layer which outputs a prediction. 
        in : input
        out : output
        """

        res = x
        for hidden_layer in self.hidden: 
            res = hidden_layer.forward_activation(res)
        self.output.forward_activation(res)
        return self.output.combi_lin

    def back_propagation(self, x, y):
        # Rétroprogration à travers la couche de sortie
        gradients = []
        gradients.append(self.output.backward_activation(y, self.hidden[-1]))

        # À travers les couches cachées
        backward = gradients[0][1].dot(self.output.W.T)
        j = 1
        for i in range(len(self.hidden)):
            layer = self.hidden[i]
            if i == 0:
                gradients.append(layer.backward_activation(backward, np.expand_dims(x, axis=0)))
            else:
                gradients.append(layer.backward_activation(backward, self.hidden[i-1].res_forward))

            backward = gradients[j][1].dot(layer.W.T)
            j+=1
        
        self.update(gradients)

    def update(self, gradients):
        # Mise à jour des couches cachées
        learning_rate_modified = self.learning_rate * math.pow((1 + (self.learning_rate * self.T)), -1)
        for i in range(len(self.hidden)):
            Wgrad, bgrad = gradients.pop()
            self.hidden[i].W -= learning_rate_modified * Wgrad
            self.hidden[i].b -= learning_rate_modified * bgrad

        # Mise à jour de la couche de sortie
        Wgrad, bgrad = gradients.pop()
        self.output.W -= learning_rate_modified * Wgrad
        self.output.b -= learning_rate_modified * bgrad

    # ---------------------- Training ---------------------

    def train(self, train, dev, epochs=1000):
        
        start_time = time()
        X, Y = train
        index4shuffle = [i for i in range(len(X))]
        for e in range(1, epochs+1, 1):
            n = 0
            # SHUFFLE
            random.shuffle(index4shuffle)
            for index in index4shuffle:
                input, output = X[index], Y[index]
                self.forward_propagation(input)
                self.output.activation()
                self.back_propagation(input, output)
                self.T += 1
                n+=1

            ### Prints
            if e % 10 == 0:
                print(f"Epoch {e}")
        print("Training time: {0:.2f} secs".format(time() - start_time))
        start_time = time()
        # We evaluate the training and devloppement data on the same length
        # Accuracy and Loss on training data
        acc_train, loss_train = self.evaluate(train[:len(dev)])
        print(acc_train)
        print(loss_train)
        # Accuracy and Loss on developpement data
        acc_dev, loss_dev = self.evaluate(dev)
        print(acc_dev)
        print(loss_dev)
        
        print("Evaluation time: {0:.2f} secs".format(time() - start_time))


    def predict(self, ex):
        return np.argmax(self.forward_propagation(ex))

    def evaluate(self, dataset):
        """ Return the number of test inputs for which the neural network ouputs the correct results """
        acc = 0.0
        loss = 0.0
        total = 0.0
        X,Y = dataset
        for x, y_gold in zip(X,Y):
            y_gold = y_gold
            probs = self.output.activation()
            loss -= np.log(probs[:,y_gold])
            y_hat = self.predict(x)
            print(y_hat, y_gold)
            if y_hat == y_gold:
                acc += 1
            total +=1

        return acc * 100 / total, loss / total


if __name__ == "__main__":
    X = [np.array(x) for x in [[0,0], [0,1], [1,0], [1,1]]]
    Y = [np.array(y) for y in [0, 1, 1 ,0]]
    train = X,Y
    NN = NeuralNetwork([50], [RELU], [0,1], 0.01)
    NN.train(train, train)
