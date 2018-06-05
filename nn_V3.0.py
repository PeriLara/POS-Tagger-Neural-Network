#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti


""" Optimizations
        matrice de confusion
        borner les valeurs des paramètres : 
            2 solutions :   normeL2 : R(w,b) = ||w,b||^2
                            normeL1 : R(w,b) = somme sur i ||wi,bi||
        Xavier --> Weight à revoir
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

RAND = "Random"
ZERO = "Zero"
XAVIER = "Xavier"

HINGE = "Hinge"
CROSS = "Cross"
CLASSIC = "Classic"

NL2 = "L2"
NL1 = "L1"


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

class Sigmoid_Layer(Layer):
    """ Layer with sigmoid activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = 0.0 #result of the forward activation
        self.combi_lin = 0.0

    def __str__(self):
        return f"Sigmoid Layer : Weights = {self.W}, {self.W.shape} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward_activation(self, x): 
        # return the derivative of sigmoid 
        return self.forward_activation(x) * (1.0 - self.forward_activation(x))

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
    
    def backward_activation(self, x):
        # return the derivative of softmax
        return self.forward_activation(x) * (1 - self.forward_activation(x))

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
        return 1.0 * (x > 0.0)

class Tanh_Layer(Layer):
    """ Layer with Tanh activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W)
        self.b = b
        self.res_forward = 0.0 #result of the forward activation
        self.combi_lin = 0.0

    def __str__(self):
        return f"Tanh Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_activation(self, x):
        return ( 2.0 / 1.0 + np.exp(-2.0*x) ) - 1.0
    
    def backward_activation(self, x):
        # return the derivative of tanH
        return 1.0 - (self.forward_activation(x) ** 2.0) 

class Lookup_Table_Layer(Layer):
    """ Different from a lambda layer, the input layer for a POS Tagger NN
        Activation function = Linear
    """
    def __init__(self, W, vocabulary, window=2):
        
        self.vocabulary = vocabulary            # list containing all the words from the training data
        self.nbr_words = len(self.vocabulary)

        self.res_forward = 0.0
        self.combi_lin = 0.0

        self.table = defaultdict()              # {word : vector one_hot}

        self.window = window                    # window for the embeddings, default = 2
        Layer.__init__(self, W)



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
        
    def generate_onehots(self):
        # generates a one_hot vector shape (1, length of vocabulary)
        for i,word in enumerate(self.vocabulary):
            onehot = np.zeros(shape=(1, self.nbr_words))
            onehot[0][i] = 1
            self.table[word] = onehot

    def average_vec(self, context): 
        return sum(context)/len(context)

    def concatenations(self, sentences, classe2index, voc2index, test=False):

        #X = np.zeros(shape=((window*2+1), len(self.vocabulary)))
        #y = np.zeros(shape=(len(classe2index),))
        X = []
        y = []
        for sentence in sentences:
            longueur_phrase = len(sentence[0])
            for i, word_tag in enumerate(zip(sentence[0], sentence[1])):
                word, tag = word_tag
                context = np.zeros(shape=((window*2), len(self.vocabulary)))
                j = 0
                if word in self.vocabulary:
                    index_word = voc2index[word]
                else:
                    index_word = voc2index[UNKNOWN]
                for w in range(-window+1, window):
                    if i + w < 0 and i != 0:                   context[j][voc2index[f"d{w}"]] = 1
                    elif i + w >= longueur_phrase and i != 0:  context[j][voc2index[f"f{w}"]] = 1
                    elif i != 0:                               context[j][index_word] = 1
                    j+=1
                context = self.average_vec(context)
                X.append(context)
                y.append(Y_onehots[classe2index[tag]])

        X = np.array(X)
        y = np.array(y)
        return X, y



# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, classes, layers, activ, vocabulary, voc2index, sentences, init=XAVIER, norme=NL2, learning_rate=0.1, cost_function=CROSS, window=2, test=None):
        """ @param : classes       list containing the classes
            @param : layers        list containing the number of neurons in each layer
            @param : activ         list containing for each layer his activation function
            @param : vocabulary    list containing all the words of the training set, finit
            @param : voc2index     dictionary {word : index in the list vocabulary}, finit
            @param : sentences     list of tuples (sentence, tags)
            @param : init          parameter's initalization, could be Random(RAND), Zeros(ZERO) 
                                        or Xaviers initialization(XAVIER) 
            @param : norme         normalize the parameters, default normeL2 = ||w,b||^2
            @param : learning_rate default=0.1
            @param : cost_function the cost function, can be Hinge Loss, Cross Entropy or 'classic'
            @param : window        window of words for the embeddings
            @param : lookup_size   dimension of the weight for the lookup table layer
            @param : test          only sentences of the test set
        """
        
        self.layers = layers
        self.learning_rate = learning_rate
        self.nbr_classes = len(classes)         
        self.classes = classes
        self.class2index = { x : i for i,x in enumerate(self.classes) }

        self.vocabulary = vocabulary
        self.voc2index = voc2index
        self.window = window
        self.sentences = sentences
        self.lookup_size = self.layers[0]
        self.cost_function = cost_function


        self.param = []
        self.init = init
        if init == ZERO:
            self.init_param_zero(activ, layers)
        elif init == RAND:
            self.init_param_random(activ, layers)
        #else:
        #    self.init_param_xavier(activ, layers)
        self.nb_param = len(self.param)


        self.X, self.y = self.param[0].concatenations(sentences, self.class2index, self.voc2index)
        if test:
            self.X_test, self.y_test= self.param[0].concatenations(test, self.class2index, self.voc2index, True)            
            self.test=True
        else: 
            self.test=False

        self.index4shuffle = [i for i in range(len(self.X))] # cf. def train 


    # ---------------------- Init parameters ---------------------

    def init_param_random(self, activ, layers):
        """
        Initializes the parameters randomly for weight 
            & to zeros for bias
        """     
        print("RANDOM INITIALIZATION") 
        bfr_W = np.random.random(size=(len(self.vocabulary), self.lookup_size )).astype(np.dtype("float64")) * 0.01
        self.param.append(Lookup_Table_Layer(bfr_W, self.vocabulary, self.window))
        self.param[0].generate_onehots()

        print(activ, layers)
        for i, activ in enumerate(activ):
            dim, next_dim = layers[i], layers[i+1]
            bfr_W = np.random.random(size=(dim, next_dim)).astype(np.dtype("float128")) * 0.01
            bfr_b = np.zeros(shape=(next_dim,1)).astype(np.dtype("float128"))


            if activ == SOFTMAX or i == len(activ)-1:
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

    def init_param_zero(self, activ, layers):
        """
        Initializes the parameters 
            to all zeros for both weights and bias
        """   
        print("ZERO INITIALIZATION")  
        bfr_W = np.zeros(shape=(len(self.vocabulary), self.lookup_size)).astype(np.dtype("float128")) * 0.01
        self.param.append(Lookup_Table_Layer(bfr_W, self.vocabulary, self.window))  
        self.param[0].generate_onehots()      
        for i, activ in enumerate(activ):
            dim, next_dim = layers[i], layers[i+1]
            bfr_W = np.zeros(shape=(dim, next_dim)).astype(np.dtype("float128")) * 0.01
            bfr_b = np.zeros(shape=(next_dim,1)).astype(np.dtype("float128"))

            if activ == SOFTMAX or i == len(activ)-1:
                self.param.append(Softmax_Layer(bfr_W, bfr_b))
            elif activ == SIGMOID:
                self.param.append(Sigmoid_Layer(bfr_W, bfr_b))
            elif activ == RELU:
                self.param.append(RelU_Layer(bfr_W, bfr_b))
            elif activ == TANH:
                self.param.append(Tanh_Layer(bfr_W, bfr_b))
            elif activ == LIN:
                self.param.append(Linear_Layer(bfr_W, bfr_b))
            else: #default
                self.param.append(Sigmoid_Layer(bfr_W, bfr_b))


    # ---------------------- Propagations ---------------------

    def forward_propagation(self, X, test=False):
        """
        Prediction of the NN.
        Input data, X,  is “forward propagated” through the network 
            layer by layer to the final layer which outputs a prediction. 
        in : input
        out : output
        """
        lutl = self.param[0]
        combi_lin = np.dot(X, lutl.W)
        res = lutl.forward_activation(combi_lin)
        lutl.res_forward = res
        lutl.combi_lin = combi_lin

        output = []
        for i,p in enumerate(self.param): 
            if i != 0:
                combi_lin = np.add(np.matmul(res, p.W), p.b.T)
                res = p.forward_activation(combi_lin)

                p.res_forward = res
                p.combi_lin = combi_lin
                output.append(res)
        return output[-1]

    def back_propagation(self, tag, p=False):
        _y = self.param[-1].res_forward
        for i in reversed(range(len(self.param))):

            param = self.param[i]
            if i == self.nb_param - 1:    # starting the backpropagation with the output layer
                error = self.cost(_y, tag)  #output error 
                slope = param.backward_activation(param.combi_lin)
                delta = error * slope

                self.param[i].W -= self.param[i-1].res_forward.T.dot(delta) * self.learning_rate
                self.param[i].b -= (np.sum(delta, keepdims=True) * self.learning_rate)[0]

            elif i == 0: 
                error = delta.dot(self.param[i+1].W.T)
                slope = param.backward_activation(param.combi_lin)
                delta = error * slope

                #update
                self.param[i].W -= self.X.T.dot(delta) * self.learning_rate

            else:
                error = delta.dot(self.param[i+1].W.T)
                slope = param.backward_activation(param.combi_lin)
                delta = error * slope

                #update
                self.param[i].W -= self.param[i-1].res_forward.T.dot(delta) * self.learning_rate
                self.param[i].b -= (np.sum(delta, keepdims=True) * self.learning_rate)[0]

    def cost(self, _y, y):
        if self.cost_function == HINGE:
            return np.max(0, 1 - _y * y)
        elif self.cost_function == CROSS:
            return -y * np.log(_y)  
        return _y - y

    # ---------------------- Training ---------------------

    def train(self, epochs=5, mini_batch_size=None):
        """ trains a neural network using the SGD or the GD algorthm """
        # if errors
        if mini_batch_size != None:
            if mini_batch_size < len(self.X):
                mini_batch_size = len(self.X)/ 2

        n = len(self.X)
        start_time = time()
        for e in range(1, epochs+1, 1):

            random.shuffle(self.index4shuffle) # shuffle indexes of the training data
            self.X = np.array([self.X[i] for i in self.index4shuffle])
            self.y = np.array([self.y[j] for j in self.index4shuffle])
            ### SGD
            if mini_batch_size: 
                list_mini_batches = np.array([ self.X[k : k+mini_batch_size] 
                                    for k in range(0, n, mini_batch_size)])
                for _ in list_mini_batches:
                    
                    self.forward_propagation(self.X)
                    self.back_propagation(self.y)

            ### GD
            else:
                self.forward_propagation(self.X)
                
                if (e%1000) == 0:
                    self.back_propagation(self.y, p=True)
                else:
                    self.back_propagation(self.y)
                    #print(f"-----------------------------{e}--------------------")
                    #print(self.param[0].W)

            
            ### Prints
            #if (e%1000) == 0:
            if self.test:
                print(f"Epoch : {e}, Evaluation : {self.evalute()/ n}")
            else:
                print(f"Epoch : {e} complete")
        
        print("Training time: {0:.2f} secs".format(time() - start_time))



    def evalute(self):
        """ Return the number of test inputs for which the neural network ouputs the correct results """
        #if self.nbr_classes > 2:    test_results = [(np.argmax(self.forward_propagation(x)), y) for x,y in zip(self.X_test, self.y_test)]
        test_results =[(np.argmax(self.forward_propagation(x)), np.argmax(y)) for x,y in zip(self.X_test, self.y_test)]
        print(self.forward_propagation(self.X_test))

        return sum(int(a==b) for a,b in test_results)


if __name__ == "__main__":

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
    train_sentences, train_voc2index, train_vocabulary = rc.read_conllu(trainfile, window)
    classes = rc.get_classes(train_sentences)
    X_onehots, Y_onehots = rc.create_onehots(train_vocabulary, classes)

    # Creating test data 
    (test_sentences, test_voc2index, test_vocabulary) = rc.read_conllu(testfile, window, train_vocabulary)


    NN = NeuralNetwork(classes, [19,60,len(classes)],[TANH, SOFTMAX], train_vocabulary, train_voc2index, train_sentences, RAND, NL2, 0.1, CLASSIC, window,test_sentences )  
    NN.train()