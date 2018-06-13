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
from abc import ABC

# ---------------------- Constants ---------------------
UNKNOWN= "UNKNOWN"

RELU = "relu"
SIGMOID = "sigmoid"
SOFTMAX = "softmax"
TANH = "tanh"
LIN = "linear"

# ---------------------- Layers ---------------------
""" Chaque layer hérite de la classe Layer
    Chaque layer a 4 attributs principaux : 
        les paramètres du réseau : la matrice de poids W et le vecteur de biais b
        res_forward : correspond au résultat de la fonction d'acivation 
        combi_lin : correspond au résultat de la combinaison linéaire avant son passage par la fonction d'activation
    Chaque layer a 3 fonctions principales : 
        __str__
        forward_activation : correspond à la fonction d'activation du layer
        backward_activation : correspond à la fonction d'activation dérivée du layer (except softmax)

    Les classes d'embedding et Lookup sont différentes des autres car elles s'occupent en plus des entrées du réseau """

class Layer(ABC): #abstract
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __repr__(self):
        return f"Layer : Weights = {self.W}"

    def __str__(self):
        return f"Layer : Weights = {self.W}"
    
    def forward_function(self, x):
        return

    def backward_function(self, x):
        return


class Linear_Layer(Layer):
    """ Layer with linear activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"Linear Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"

    def __str__(self):
        return f"Linear Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x, self.W), self.b.T)
        self.res_forward = self.combi_lin
        return self.res_forward
    
    def backward_function(self, backward): # returns the derivative of linear function
        Wgrad = self.res_forward.T.dot(backward)
        return backward, Wgrad


class Softmax_Layer(Layer):
    """ Layer with sigmoid activation function 
        Always the output layer """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"Softmax Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"

    def __str__(self):
        return f"Softmax Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x,self.W), self.b.T)
        new_x = self.combi_lin - np.max(self.combi_lin)
        exp = np.exp(new_x)
        self.res_forward = exp / exp.sum()
        return self.res_forward

    def backward_function(self, y, hiddenL): 
        new_x = self.combi_lin - np.max(self.combi_lin)
        exp = np.exp(new_x)
        backward = (exp/ exp.sum()) - y # c'est le gradient par rapport à cross_entropy(softmax)
        Wgrad = hiddenL.res_forward.T.dot(backward) #slope
        return backward, Wgrad


class Sigmoid_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"Sigmoid Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def __str__(self):
        return f"Sigmoid Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x,self.W), self.b.T)
        self.res_forward = 1.0 / 1.0 + np.exp(-self.combi_lin)
        return self.res_forward
    
    def backward_function(self, backward): 
        backward = backward.dot(self.combi_lin * (1.0 - self.combi_lin))
        Wgrad = self.res_forward.T.dot(backward)
        return backward, Wgrad


class RelU_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    

    def __str__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x,self.W), self.b.T)
        self.res_forward = self.combi_lin * (self.combi_lin > 0.0)
        return self.res_forward
    
    def backward_function(self, backward): #returns the jacobian of the function
        #jac = np.diag((1.0 * (x > 0.0))[0])
        backward = backward.dot(1.0 * (self.combi_lin > 0.0))
        Wgrad = self.res_forward.T.dot(backward)
        return backward, Wgrad


class Tanh_Layer(Layer):
    """ Layer with TanH activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"Tanh Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def __str__(self):
        return f"Tanh Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x,self.W), self.b.T)
        self.res_forward = (2.0 / 1.0 + np.exp(-2.0 * self.combi_lin) ) * -1.0
        return self.res_forward
    
    def backward_function(self, backward): 
        backward = backward.dot(1.0 - (self.combi_lin ** 2.0))
        Wgrad = self.res_forward.T.dot(backward)
        return backward, Wgrad


class Embedding_Layer(Layer):
    """ Layer with RelU activation function 
        En plus des attributs communs aux Layers
            input_vectors : liste des vecteurs denses extraits de Lookup, se propageant dans le réseau par concaténation
            input_names : liste des noms des mots correspondants aux vecteurs denses, 
                        utile quand on veut modifier après retropropagation dans le Lookup Layer
            window : fenêtre de mots à prendre en compte dans le contexte
        En plus des fonctions communes aux Layers
            set_features : prend l'indice d'un mot et la phrase dans laquelle il se trouve, 
                            remplie les attributs d'inputs avec les vecteurs de contexte et le vecteur du mot
        """
    def __init__(self, W, b, window):
        Layer.__init__(self, W, b)
        self.res_forward = [] #result of the forward activation
        self.combi_lin = []
        self.inputs_vectors = []
        self.inputs_names = []
        self.window = window

    def __str__(self):
        return f"Embedding Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def forward_function(self, sentence, x, table, voc):
        self.set_features(sentence, x, table, voc)
        for mot in self.inputs_vectors:
            combi_lin = np.add(np.matmul(mot, self.W), self.b.T)
            self.combi_lin.append(combi_lin)
            self.res_forward.append(combi_lin)
        conc = np.concatenate(self.res_forward, axis=1)
        return conc

    def backward_function(self, backward): # returns the derivative of linear function
        error = np.split(backward, self.window*2+1, axis=1) 
        gradients = []
        for i in range(len(error)):
            backward = error[i]
            Wgrad = self.res_forward[i].T.dot(backward) 
            Wgrad = Wgrad.dot(self.W.T) 
            essai = error[i].dot(self.W.T)
            gradients.append((backward, Wgrad, essai))
        return gradients #len = window*2+1


    def set_features(self, sentence, i_word, table, vocabulary):
        for w in range(-self.window, self.window+1):

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

    def update(self):
        self.inputs_vectors = []
        self.inputs_names = []
        self.res_forward = []            
        self.combi_lin = []


class Lookup_Layer(Layer):
    """
    En plus des attributs communs aux Layers
        vocabulary :    liste du vocabulaire de l'ensemble d'entrainement (avec Unknown et les mots de contexte en dehors de la phrase)
        voc2index :     dictionnaire word:index
        table :         dictionnaire word : dense vector
        conc :          reçoit la concaténation des vecteurs denses activés par l'Embedding Layer
    En plus des fonctions communes aux Layers
        get_vector      retourne le vecteur correspondant au mot 
    """
    def __init__(self, W,b, vocabulary, embedding):
        Layer.__init__(self, W, b)
        self.res_forward = None
        self.combi_lin  = None

        self.vocabulary = vocabulary
        self.voc2index = {x:i for i,x in enumerate(self.vocabulary)}

        self.table = defaultdict(lambda:np.random.rand((1,embedding.W.shape[0])))
        # On remplie la table
        self.table = {word : np.random.random(size=(1, embedding.W.shape[0])) for word in self.vocabulary}

        self.conc = None

    def __str__(self):
        return f"Lookup Layer : Weights = {self.W} \n activation = {self.res_forward}"
    
    def forward_function(self, x): #Linear
        self.combi_lin = np.add(np.matmul(x, self.W), self.b.T)
        self.res_forward = self.combi_lin
        return self.res_forward
    
    def backward_function(self, backward): # Linear
        Wgrad = self.conc.T.dot(backward) 
        return backward, Wgrad

    def get_vector(self, word): # returns from the LookupTable the vector of the word
        if word not in self.table:  word=UNKNOWN
        return self.table[word]
    
    def update(self):
        self.conc = []

# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, layers, activ, vocabulary, window, classes, embedding=10, learning_rate=0.1):  
        """ Neural Network 
            @param layers       liste du nombre de neurones par layer
            @param activ        liste des fonctions d'activation pour chaque layer
            @param vocabulary   vocabulaire du set d'entrainement
            @param window       fenetre de contexte
            @param classes      liste des classes / tags
            @param embedding    nombre de neurones du layer d'embedding
            @param learning_rate default=0.1
        """
        
        self.layers = layers
        self.learning_rate = learning_rate
        self.T = 1 # optimization with learning rate

        self.vocabulary = vocabulary
        self.window = window

        self.embedding_size = embedding
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
        # Intiailization of the Embedding Layer
        bfr_W = np.random.uniform((-(math.sqrt(6) / math.sqrt(self.embedding_size + layers[0]))), (math.sqrt(6) / math.sqrt(self.embedding_size + layers[0])), (self.embedding_size, layers[0]))
        bfr_b = np.random.uniform((-(math.sqrt(6) / math.sqrt(layers[0]))), (math.sqrt(6) / math.sqrt(layers[0])), ( layers[0], 1))
        self.embedding = Embedding_Layer(bfr_W, bfr_b, self.window)    

        for i, activ in enumerate(activ):
            if i == 0: #Lookup Layer
                bfr_W = np.random.uniform(((-(math.sqrt(6) / math.sqrt((layers[i]*(self.window*2+1)) + layers[i+1])))), (math.sqrt(6) / math.sqrt(layers[i]*(self.window*2+1)) + layers[i+1]), (layers[i]*(self.window*2+1), layers[i+1]))
            else:
                mini = -(math.sqrt(6) / (math.sqrt(layers[i] + layers[i+1])))
                maxi = math.sqrt(6) / math.sqrt(layers[i] + layers[i+1])
                bfr_W = np.random.uniform(mini, maxi, (layers[i], layers[i+1]))
            bfr_b = np.random.uniform((-(math.sqrt(6) / math.sqrt(layers[i+1]))), (math.sqrt(6) / math.sqrt(layers[i+1])), ( layers[i+1], 1))


            if i == 0:
                self.param.append(Lookup_Layer(bfr_W, bfr_b, self.vocabulary, self.embedding))
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
        For each word x in the sentence
        pred = False --> training set
        pred = True --> test set
        """
        # Forward propagation for the Embedding Layer which is not fully connected
        res = self.embedding.forward_function(sentence, x, self.param[0].table, self.param[0].vocabulary)
        self.param[0].conc = res
        
        # Forward propagation for other layers which are fully connected
        for p in self.param: 
            res = p.forward_function(res)

        if pred:
            self.embedding.update()
            self.param[0].update()


        return self.param[-1].res_forward # output of the forward prop

    def back_propagation(self, y):
        # Backprop for the output layer, different from the others
        backward, Wgrad = self.param[-1].backward_function(y, self.param[-2])
        assert Wgrad.shape == self.param[-1].W.shape
        assert backward.shape == self.param[-1].b.T.shape
        gradients = [(Wgrad, backward)]
        backward = backward.dot(self.param[-1].W.T) #(1,60)

        # Backprop for hiddenlayers + lookuplayer
        for i in reversed(range(len(self.param) - 1)):
            param = self.param[i]
            backward, Wgrad = param.backward_function(backward) # avant i-1
            assert backward.shape == param.b.T.shape
            assert Wgrad.shape == param.W.shape
            gradients.append((Wgrad, backward)) 
            backward = backward.dot(param.W.T) 

        # EMBEDDING PART
        emb_gradients = self.embedding.backward_function(backward)

        self.update_params(gradients, emb_gradients)

        self.embedding.update()
        self.param[0].update()

    def update_params(self, layers_gradients, emb_gradients):
        # Update from output to lookup layer
        learning_rate_modified = self.learning_rate * math.pow((1 + (self.learning_rate * self.T)), -1)
        for i in range(len(self.param)):
            Wgrad, bgrad = layers_gradients.pop()
            self.param[i].W -= learning_rate_modified * Wgrad
            self.param[i].b -= learning_rate_modified * bgrad.T

            self.param[i].W = (self.param[i].W - self.param[i].W.min()) / (self.param[i].W.max() -self.param[i].W.min())
            self.param[i].b = (self.param[i].b - self.param[i].b.min()) / (self.param[i].b.max() -self.param[i].b.min())
        
        #Update Embeddings
        for e in range(len(emb_gradients)-1, -1, -1):
            bgrad, Wgrad, igrad = emb_gradients.pop(e)
            
            self.embedding.W -= learning_rate_modified * Wgrad.T
            self.embedding.b -= learning_rate_modified * bgrad.T
            self.param[0].table[self.embedding.inputs_names[e]] -=  learning_rate_modified * igrad

            self.embedding.W = (self.embedding.W - self.embedding.W.min()) / (self.embedding.W.max() -self.embedding.W.min())
            self.embedding.b = (self.embedding.b - self.embedding.b.min()) / (self.embedding.b.max() -self.embedding.b.min())
            


    # ---------------------- Training ---------------------

    def train(self, sentences, test_sentences=None, epochs=20, mini_batch=None):
        start_time = time()

        for e in range(1, epochs+1, 1):
            random.shuffle(sentences) # shuffle indexes of the training data

            """if mini_batch: #SGD
                n = len(sentences)
                assert n > mini_batch
                mini_sentences = np.array([ sentences[k : k+mini_batch] 
                                                for k in range(0, n, mini_batch)])
                for sentence in mini_sentences:
                    print(sentence)
                    for i in range(len(sentence)):
                        self.forward_propagation(sentence, i)
                        self.back_propagation(self.classe2one_hot[sentence[1][i]])"""
            for sentence in sentences:
                for i in range(len(sentence)):
                    self.forward_propagation(sentence, i)
                    self.back_propagation(self.classe2one_hot[sentence[1][i]])
                    self.T+=1
            ### Prints
            if e % 1 == 0:
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
    parser.add_argument("--mini_batch_size","-mbs", type=int, default=None, help="Dimension of the weight for the lookup table layer")
    args = parser.parse_args()

    trainfile = args.train
    # devfile = args.dev
    testfile = args.test
    window = args.window
    
    # Creating training data
    train_sentences, train_vocabulary = rc.read_conllu(trainfile, window)
    classes = rc.get_classes(train_sentences)

    # Creating test data 
    test_sentences, test_vocabulary = rc.read_conllu(testfile, window, train_vocabulary)

    NN = NeuralNetwork([12,20,len(classes)],[RELU, SOFTMAX], train_vocabulary, window, classes)  
    NN.train(train_sentences, test_sentences=test_sentences, mini_batch=args.mini_batch_size)
    #print(NN.evalute())