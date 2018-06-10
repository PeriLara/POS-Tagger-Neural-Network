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
        self.res_forward = [] #result of the forward activation
        self.combi_lin = []

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
        self.res_forward = None #result of the forward activation
        self.combi_lin = None


    def __str__(self):
        return f"Softmax Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

class RelU_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __str__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return x * (x > 0.0)
    
    def backward_activation(self, x):
        jac = np.diag((1.0 * (x > 0.0))[0])
        return jac

class Embedding_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = [] #result of the forward activation
        self.combi_lin = []
        self.inputs_vectors = []
        self.inputs_names = []

    def __str__(self):
        return f"Embedding Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return x 
    
    def backward_activation(self, x):
        # return the derivative of RelU
        return 1.0

    def set_features(self, sentence, i_word, table, vocabulary):
        for w in range(-window, window+1):

            if i_word + w < 0:
                self.inputs_vectors.append(table[f"d{w}"])
                self.inputs_names.append(f"d{w}")
            elif i_word + w >= len(sentence):  
                self.inputs_vectors.append(table[f"f{w}"])
                self.inputs_names.append(f"f{w}")
            else:                                      
                if sentence[i_word + w] in vocabulary: 
                    self.inputs_vectors.append(table[sentence[i_word]])
                    self.inputs_names.append(sentence[i_word])

                else:        
                    self.inputs_vectors.append(table[UNKNOWN])
                    self.inputs_names.append(UNKNOWN)

class Lookup_Layer(Layer):
    def __init__(self, W,b, vocabulary, window, embedding):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = None
        self.combi_lin  = None

        self.vocabulary = vocabulary
        self.voc2index = {x:i for i,x in enumerate(self.vocabulary)}
        self.window = window

        self.table = defaultdict()
        self.table = {word : np.random.random(size=(1, embedding.W.shape[0])) for word in self.vocabulary}

        self.conc = None

    def __str__(self):
        return f"Lookup Layer : Weights = {self.W} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        self.res_forward = x
        return x
    
    def backward_activation(self, x):
        # return the derivative of a linear function
        return 1.0 

    def get_vector(self, word):
        # return from the LookupTable the vector of the word
        if word not in self.table:  word=UNKNOWN
        return self.table[word]






# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, layers, activ, vocabulary, window, classes, learning_rate=0.1):  
        self.layers = layers
        self.learning_rate = learning_rate

        self.vocabulary = vocabulary
        self.window = window


        self.param = []
        self.init_param_random_xavier(activ, layers)
        self.nb_param = len(self.param)

        self.classe2one_hot = defaultdict()
        for i, classe in enumerate(classes):
            one_hot = np.zeros(shape=(1, len(classes)))
            one_hot[0][i] = 1
            self.classe2one_hot[classe] = one_hot



    # ---------------------- Init parameters ---------------------

    def init_param_random_xavier(self, activ, layers):
        """
        Initializes the parameters randomly for weight 
            & to zeros for bias
        """ 
        bfr_W = np.random.random(size=(40, layers[0]))
        bfr_W -= 0.5
        bfr_W /= 100
        bfr_b = np.random.random(size=(layers[0],1)) 
        bfr_b -= 0.5
        bfr_b /= 100
        self.embedding = Embedding_Layer(bfr_W, bfr_b)    
        for i, activ in enumerate(activ):
            if i == 0:
                bfr_W = np.random.random(size= (layers[i]*(self.window*2+1), layers[i+1]))
            else:
                bfr_W = np.random.random(size=(layers[i], layers[i+1]))
            bfr_W -= 0.5
            bfr_W /= 100
            bfr_b = np.random.random(size=(layers[i+1],1)) 
            bfr_b -= 0.5
            bfr_b /= 100

            if i == 0:
                self.param.append(Lookup_Layer(bfr_W, bfr_b, self.vocabulary, self.window, self.embedding))
            elif activ == SOFTMAX:
                self.param.append(Softmax_Layer(bfr_W, bfr_b))
            elif activ == RELU:
                self.param.append(RelU_Layer(bfr_W, bfr_b))
            elif activ == LIN:
                self.param.append(Linear_Layer(bfr_W, bfr_b))
            else: # default
                self.param.append(Linear_Layer(bfr_W, bfr_b))



    # ---------------------- Propagations ---------------------

    def forward_propagation(self, sentence, x, pred=False):
        """
        Prediction of the NN.
        Input data, X,  is “forward propagated” through the network 
            layer by layer to the final layer which outputs a prediction. 
        in : input
        out : output
        """


        self.embedding.set_features(sentence, x, self.param[0].table, self.param[0].vocabulary)
        for mot in self.embedding.inputs_vectors:
            combi_lin = np.add(np.matmul(mot, self.embedding.W), self.embedding.b.T)
            self.embedding.combi_lin.append(combi_lin)
            self.embedding.res_forward.append(self.embedding.forward_activation(combi_lin))

        res = np.concatenate(self.embedding.res_forward, axis=1)
        self.param[0].conc = res
        for p in self.param: 
            combi_lin = np.add(np.matmul(res, p.W), p.b.T)
            p.combi_lin = combi_lin

            res = p.forward_activation(combi_lin)
            p.res_forward = res

        if pred:
            self.embedding.inputs_vectors = []
            self.embedding.inputs_names = []
            self.embedding.res_forward = []
            self.embedding.combi_lin = []
            self.param[0].conc = []

        return res


    def back_propagation(self, y):
        backward = self.param[-1].forward_activation(self.param[-1].combi_lin) - y # c'est le gradient par rapport à cross_entropy(softmax)
        Wgrad = self.param[-2].res_forward.T.dot(backward) #slope

        assert Wgrad.shape == self.param[-1].W.shape
        assert backward.shape == self.param[-1].b.T.shape
        gradients = [(Wgrad, backward)]
        backward = backward.dot(self.param[-1].W.T) #(1,60)

        for i in reversed(range(len(self.param) - 1)):
            param = self.param[i]
            backward = backward.dot(param.backward_activation(param.combi_lin)) ## error, (1,60)

            if i == 0:
                Wgrad = self.param[0].conc.T.dot(backward) ## delta 95,60
                
            else:
                Wgrad = self.param[i-1].res_forward.T.dot(backward) ## delta (1,40)


            assert backward.shape == param.b.T.shape
            assert Wgrad.shape == param.W.shape
            gradients.append((Wgrad, backward)) 
            backward = backward.dot(param.W.T) #(1,95)

        for i in range(len(self.param)):
            Wgrad, bgrad = gradients.pop()
            self.param[i].W -= self.learning_rate * Wgrad
            self.param[i].b -= self.learning_rate * bgrad.T

            self.param[i].W = (self.param[i].W - self.param[i].W.min()) / (self.param[i].W.max() -self.param[i].W.min())


        #### EMBEDDING PART
        error = np.split(backward, 5.0, axis=1) # 5 * (1,19)
        for i in range(len(error)):
            e = error[i].dot(self.embedding.backward_activation(self.embedding.combi_lin[i])) #(1,19)
            Wgrad = self.embedding.res_forward[i].T.dot(e) #( 19,19)
            Wgrad = Wgrad.dot(self.embedding.W.T) #19,40
            essai = error[i].dot(self.embedding.W.T)
            
            gradients.append((Wgrad, e, essai))

        for e in range(len(error)-1, -1, -1):
            Wgrad, bgrad, igrad = gradients.pop(e)
            self.embedding.W -= self.learning_rate * Wgrad.T
            self.embedding.b -= self.learning_rate * bgrad.T
            self.param[0].table[self.embedding.inputs_names[e]] -=  self.learning_rate * igrad


        
        self.embedding.inputs_vectors = []
        self.embedding.inputs_names = []
        self.embedding.res_forward = []
        self.embedding.combi_lin = []
        self.param[0].conc = []


    # ---------------------- Training ---------------------

    def train(self, sentences, test_sentences=None, epochs=20):
        start_time = time()
        for e in range(1, epochs+1, 1):
            for sentence in sentences:
                for i in range(len(sentence)):
                    self.forward_propagation(sentence, i)
                    self.back_propagation(self.classe2one_hot[sentence[1][i]])

            ### Prints
            if e % 10 == 0:
                print("Epoch : {}, Evaluation : {}".format(e, self.evaluate(test_sentences)))

        print("Training time: {0:.2f} secs".format(time() - start_time))


    def predict(self, sentence, i_word):
        #print("PREDICT", self.forward_propagation(sentence, i_word, pred=True))
        return np.argmax(self.forward_propagation(sentence, i_word, pred=True))

    def evaluate(self, test_sentences):
        """ Return the number of test inputs for which the neural network ouputs the correct results """
        ex_nb = len(test_sentences)
        predictions = []
        Y = []
        for sentence in test_sentences:
            for i in range(len(sentence)):
                predictions.append(self.predict(sentence, i))
                Y.append(self.classe2one_hot[sentence[1][i]])
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

    NN = NeuralNetwork([19,60,len(classes)],[TANH, SOFTMAX], train_vocabulary, window, classes)  
    NN.train(train_sentences, test_sentences=test_sentences)
    #print(NN.evalute())