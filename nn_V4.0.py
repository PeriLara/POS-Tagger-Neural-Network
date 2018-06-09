#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti


""" Optimizations
        matrice de confusion
        Adam
        Momentum

    TESTS
        plusieurs learning_rates
        plusieurs tailles de hidden layers


    OPTIONS
        de shuffle
"""

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
SIGMOID = "sigmoid"
SOFTMAX = "softmax"
TANH = "tanh"
LIN = "linear"

# ---------------------- Layers ---------------------

class Layer(): #abstract
    def __init__(self, W):
        self.W = W
    
    def __str__(self):
        return f"Layer : Weights = {self.W}"

class Linear_Layer(Layer):
    """ Layer with linear activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = 0.0 #result of the forward activation
        self.combi_lin = 0.0

    def __str__(self):
        return f"Linear Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return x
    
    def backward_activation(self, x):
        # return the derivative of linear function
        return 1.0

class Softmax_Layer(Layer):
    """ Layer with sigmoid activation function 
        Always the output layer """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = 0.0 #result of the forward activation
        self.combi_lin = 0.0


    def __str__(self):
        return f"Softmax Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        max_x = x - np.max(x) #with max, more stable
        exp = np.exp(max_x)
        return exp / np.sum(exp)

class RelU_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = 0.0 #result of the forward activation
        self.combi_lin = 0.0

    def __str__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return x * (x > 0.0)
    
    def backward_activation(self, x):
        # return the derivative of RelU
        jac = np.diag((1.0 * (x > 0.0))[0])
        return jac

class Lookup_Layer(Layer):
    def __init__(self, W, vocabulary, window):
        Layer.__init__(self, W)
        self.res_forward = 0.0
        self.combi_lin  = 0.0

        self.table = defaultdict()


        self.vocabulary = vocabulary
        self.voc2index = {x:i for i,x in enumerate(self.vocabulary)}
        self.window = window




# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, layers, activ, vocabulary, window, learning_rate=0.1):  
        self.layers = layers
        self.learning_rate = learning_rate

        self.param = []
        self.init_param_random_xavier(activ, layers)
        self.nb_param = len(self.param)

        self.vocabulary = vocabulary
        self.window = window




    # ---------------------- Init parameters ---------------------

    def init_param_random_xavier(self, activ, layers):
        """
        Initializes the parameters randomly for weight 
            & to zeros for bias
        """     
        
        for (input_dim, output_dim), activ in zip(layers, activ):
            bfr_W = np.random.random(size=(input_dim, output_dim))
            bfr_W -= 0.5
            bfr_W /= 100
            bfr_b = np.random.random(size=(output_dim,1)) 
            bfr_b -= 0.5
            bfr_b /= 100

            if activ == SOFTMAX:
                self.param.append(Softmax_Layer(bfr_W, bfr_b))
            elif activ == SIGMOID:
                self.param.append(Sigmoid_Layer(bfr_W, bfr_b))
            elif activ == RELU:
                self.param.append(RelU_Layer(bfr_W, bfr_b))
            elif activ == TANH:
                self.param.append(Tanh_Layer(bfr_W, bfr_b))
            elif activ == LIN:
                self.param.append(Linear_Layer(bfr_W, bfr_b))
            else: # default
                self.param.append(Sigmoid_Layer(bfr_W, bfr_b))



    # ---------------------- Propagations ---------------------

    def forward_propagation(self, X):
        """
        Prediction of the NN.
        Input data, X,  is “forward propagated” through the network 
            layer by layer to the final layer which outputs a prediction. 
        in : input
        out : output
        """

        res = X
        for p in self.param: 
                combi_lin = np.add(np.matmul(res, p.W), p.b.T)
                p.combi_lin = combi_lin
                res = p.forward_activation(combi_lin)
                p.res_forward = res

        return res

    def back_propagation(self, x, y):
        backward = self.param[-1].forward_activation(self.param[-1].combi_lin) - y # c'est le gradient par rapport à cross_entropy(softmax)
        Wgrad =  self.param[-2].res_forward.T.dot(backward) #slope
        assert Wgrad.shape == self.param[-1].W.shape
        assert backward.shape == self.param[-1].b.T.shape
        gradients = [(Wgrad, backward)]
        backward = backward.dot(self.param[-1].W.T)

        for i in reversed(range(len(self.param) - 1)):
            param = self.param[i]
            backward = backward.dot(param.backward_activation(param.combi_lin))

            if i == 0:
                Wgrad = np.expand_dims(x, axis=1).dot(backward)
            else:
                assert False
                Wgrad = self.param[i-1].res_forward.T.dot(backward)

            assert backward.shape == param.b.T.shape
            assert Wgrad.shape == param.W.shape
            gradients.append((Wgrad, backward))
            backward = backward.dot(param.W.T)

        for i in range(len(self.param)):
            Wgrad, bgrad = gradients.pop()
            if i == 1:
                pass#print(bgrad)
            self.param[i].W -= self.learning_rate * Wgrad
            self.param[i].b -= self.learning_rate * bgrad.T


    # ---------------------- Training ---------------------

    def train(self, X, Y, epochs=1000):
        start_time = time()
        for e in range(1, epochs+1, 1):
            for input, output in zip(X, Y):
                self.forward_propagation(input)
                self.back_propagation(input, output)

            ### Prints
            if e % 10 == 0:
                print("Epoch : {}, Evaluation : {}".format(e, self.evaluate(X, Y)))

        
        print("Training time: {0:.2f} secs".format(time() - start_time))


    def predict(self, ex):
        return np.argmax(self.forward_propagation(ex))

    def evaluate(self, X, Y):
        """ Return the number of test inputs for which the neural network ouputs the correct results """
        ex_nb = len(X)
        predictions = [self.predict(ex) for ex in X]
        return sum(int(a==np.argmax(b)) for a, b in zip(predictions, Y)) * 100 / ex_nb


if __name__ == "__main__":
    """X = [np.array(x) for x in [[0,0], [0,1], [1,0], [1,1]]]
    Y = [np.array(y) for y in [[1,0], [0,1], [0,1], [1,0]]]

    NN = NeuralNetwork([(2, 8), (8, 2)],[RELU, SOFTMAX], )
    print(NN.evaluate(X, Y))
    NN.train(X, Y)
    print(NN.evaluate(X, Y))
    """

    usage = "Neural Network POS tagger"
    parser = argparse.ArgumentParser(description = usage)
    parser.add_argument("train", type = str, help="Training set - Conll-U format")
    #parser.add_argument("dev", type = str, help="Development set - Conll-U format")
    parser.add_argument("test", type = str, help="Test set - Conll-U format")
    parser.add_argument('window', type = int, help = "la fenêtre de features")
    parser.add_argument("--learningrate","-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs","-i", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lookup_size","-lus", type=int, default=16, help="Dimension of the weight for the lookup table layer")
    args = parser.parse_args()

    trainfile = args.train
    #devfile = args.dev
    testfile = args.test
    window = args.window
    
    #Creating training data
    train_sentences, train_vocabulary = rc.read_conllu(trainfile, window)
    classes = rc.get_classes(train_sentences)

    # Creating test data 
    test_sentences, test_vocabulary = rc.read_conllu(testfile, window, train_vocabulary)


    NN = NeuralNetwork(classes, [(len(train_vocabulary),19),(19,60),(60,len(classes))],[TANH, SOFTMAX], window, train_vocabulary)  
    NN.train()
    print(NN.evalute())