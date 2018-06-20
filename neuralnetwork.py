#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti

import numpy as np
import math
import random
import layers as l


# ---------------------- Constants ---------------------
RELU = "relu"
SOFTMAX = "softmax"
LIN = "linear"
TANH = 'tanh'
LOOKUP = "lookup"
# ---------------------- Neural Network ---------------------

class NeuralNetwork():
    def __init__(self, hidden_layers, hidden_activations, vocabulary, window_size, classes, embedding_size, learning_rate, xavier=True):
        """ Neural Network 
            @param hidden_layers            liste du nombre de neurones par layer
            @param hidden_activations       liste des fonctions d'activation pour chaque layer
            @param vocabulary               vocabulaire du set d'entrainement
            @param window                   fenetre de contexte, si window=2 : wi-2, wi-1, wi, wi+1, wi+2
            @param classes                  liste des classes / tags
            @param embedding_size           nombre de neurones du layer d'embedding
            @param learning_rate            default=0.01
            @param xavier                   initialiser les paramètres selon la méthode de xavier 
        """
        self.learning_rate = learning_rate
        self.T = 1 #optimization of the learning rate, set to 1
        
        self.vocabulary = vocabulary
        self.embedding_size = embedding_size
        self.window_size = window_size*2+1
        self.hidden_number = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations
        self.classes = classes
        
        self.input_size = self.embedding_size * self.window_size
        self.lookup = self.init_lookup_layer(vocabulary, embedding_size)
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
            return l.Softmax_Layer(bfr_W, bfr_b)
        elif activ == RELU:
            return l.RelU_Layer(bfr_W, bfr_b)
        elif activ == LIN:
            return l.Linear_Layer(bfr_W, bfr_b)
        elif activ == TANH:
            return l.Tanh_Layer(bfr_W, bfr_b)

    def init_lookup_layer(self, voc, embedding_size):
        return {word: (np.random.rand(1, embedding_size) - 0.5) / 50 for word in voc}

    # ---------------------- Propagations ---------------------

    def forward_propagation(self, x):
        """
        Prediction of the NN.
        Input data, X,  is “forward propagated” through the network 
            layer by layer to the final layer which outputs a prediction. 
        in : input
        out : output
        """
        res = np.concatenate([self.lookup[word] for word in x], axis=1)
        self.activated_word = res
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
        for i in reversed(range(len(self.hidden))):
            layer = self.hidden[i]
            if i == 0:
                gradients.append(layer.backward_activation(backward, self.activated_word))
            else:
                gradients.append(layer.backward_activation(backward, self.hidden[i-1].res_forward))
            backward = gradients[j][1].dot(layer.W.T)
            j+=1
        
        self.update(gradients, backward, x)

    def update(self, gradients, backward,x):
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

        # Mise à jour du dictionnaire d'embeddings   
        backward = np.split(backward, self.window_size, axis=1)
        for i, word in enumerate(x):
            self.lookup[word] -= learning_rate_modified * backward[i]
            
    # ---------------------- Training ---------------------

    def train(self, train, dev, epochs):
        start_time = time()
        X, Y = train
        index4shuffle = [i for i in range(len(X))]
        for e in range(1, epochs + 1):
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
            if n%5 ==0:
                print(f"Epoch {e}")
        print("Training time: {0:.2f} secs".format(time() - start_time))

        
        # We evaluate the training and devloppement data on the same length
        # Accuracy and Loss on training data
        acc_train, loss_train = self.evaluate(train[:len(dev)])
        print(acc_train)
        print(loss_train)
        # Accuracy and Loss on developpement data
        acc_dev, loss_dev = self.evaluate(dev)
        print(acc_dev)
        print(loss_dev)


    def predict(self, x):
        return np.argmax(self.forward_propagation(x))

    def evaluate(self, dataset): # error function
        """ Return the number of test inputs for which the neural network ouputs the correct results """
        acc = 0.0
        loss = 0.0
        total = 0.0
        X,Y = dataset
        for x, y_gold in zip(X,Y):
            y_gold = np.argmax(y_gold)
            probs = self.output.activation()
            loss -= np.log(probs[:,y_gold])
            y_hat = self.predict(x)
            if y_hat == y_gold:
                acc += 1
            total +=1

        return acc * 100 / total, loss / total
        """X, Y = dataset
        ex_nb = len(X)
        predictions = [self.predict(ex) for ex in X]
        return sum(int(a==np.argmax(b)) for a, b in zip(predictions, Y)) * 100 / ex_nb
        """