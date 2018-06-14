#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti

from abc import ABC
import numpy as np
from collections import defaultdict

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
        assert W.shape[1] == b.shape[0]
        self.W = W
        self.b = b

    def __repr__(self):
        return f"Layer : Weights = {self.W}"

    def __str__(self):
        return f"Layer : Weights = {self.W}"
    
    def forward_function(self, x):
        self.combi_lin = np.add(np.matmul(x, self.W), self.b.T)
        self.res_forward = self.non_linearity()
        assert self.combi_lin.shape == self.res_forward.shape
        return self.res_forward

    def backward_function(self, backward):
        print("LINEAR BACKWARD", backward.shape)
        bgrad = self.backprop(backward)
        print(bgrad.shape)
        Wgrad = self.res_forward.T.dot(bgrad)
        print(Wgrad.shape)
        backward = backward.dot(self.W.T) 
        print(backward.shape)
        return Wgrad, bgrad, backward


class Linear_Layer(Layer):
    """ Layer with linear activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"Linear Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"

    def __str__(self):
        return "Linear"
        return f"Linear Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def non_linearity(self):
        return self.combi_lin
    
    def backprop(self, backward): # returns the derivative of linear function
        return backward


class Softmax_Layer(Layer):
    """ Layer with sigmoid activation function 
        Always the output layer """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __str__(self):
        return "Softmax"
        #return f"Softmax Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def non_linearity(self):
        exp = np.exp(self.combi_lin - np.max(self.combi_lin))
        return exp / exp.sum()

    def backward_function(self, y, hiddenL):
        assert y.shape == self.res_forward.shape
        bgrad = (self.res_forward) - y 
        Wgrad = hiddenL.res_forward.T.dot(bgrad) #slope
        backward = bgrad.dot(self.W.T) 
        return Wgrad, bgrad, backward


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
    
    def non_linearity(self):
        return 1.0 / 1.0 + np.exp(-self.combi_lin)
    
    def backprop(self, backward): 
        return backward.dot(self.combi_lin * (1.0 - self.combi_lin))


class RelU_Layer(Layer):
    """ Layer with RelU activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None

    def __repr__(self):
        return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def __str__(self):
        return "Relu"
        #return f"RelU Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def non_linearity(self):
        return self.combi_lin * (self.combi_lin > 0.0)
    
    def backprop(self, backward):
        relu = 1.0 * (self.combi_lin > 0.0)
        return backward.dot(relu)


class Tanh_Layer(Layer):
    """ Layer with TanH activation function """
    def __init__(self, W, b):
        Layer.__init__(self, W, b)
        self.res_forward = None #result of the forward activation
        self.combi_lin = None
    
    def __str__(self):
        return "Tanh"
        #return f"Tanh Layer : Weights = {self.W} \n biais = {self.b} \n activation = {self.res_forward}"
    
    def non_linearity(self):
        return (2.0 / 1.0 + np.exp(-2.0 * self.combi_lin) ) * -1.0
    
    def backprop(self, backward):
        print("backprop tanh",self.combi_lin.shape)
        return backward.dot(1.0 - (self.combi_lin.T ** 2.0))


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
        return "Lookup"
        #return f"Lookup Layer : Weights = {self.W} \n activation = {self.res_forward}"
    
    def non_linearity(self): #Linear
        return self.combi_lin
    
    def backprop(self, backward): # Linear
        return backward

    def get_vector(self, word): # returns from the LookupTable the vector of the word
        if word not in self.table:  word=UNKNOWN
        return self.table[word]
    
    def update(self):
        self.conc = []

