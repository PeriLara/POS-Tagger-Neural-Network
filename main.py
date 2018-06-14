#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Réseau de neurones POS Tagger
# code : Lara Perinetti

""" Main du réseau neuronal """


import argparse
import read_connl as rc
from neural_network import NeuralNetwork


RELU = "relu"
SIGMOID = "sigmoid"
SOFTMAX = "softmax"
TANH = "tanh"
LIN = "linear"


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

    NN = NeuralNetwork([12,2,8,len(classes)],[TANH, SOFTMAX], train_vocabulary, window, classes)  
    NN.train(train_sentences, test_sentences=test_sentences, mini_batch=args.mini_batch_size)