#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Couches du réseau de neurones
# code : Lara Perinetti

import argparse
import numpy as np
import read_connl_1 as rc
import neuralnetwork as nn


# ---------------------- Constants ---------------------
RELU = "relu"
SOFTMAX = "softmax"
LINEAR = "linear"
TANH = 'tanh'

# ---------------------- MAIN ---------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str, help="Training set - Conll-U format")
    parser.add_argument("dev", type=str, help="Development set - Conll-U format")
    parser.add_argument("test", type=str, help="Test set - Conll-U format")
    parser.add_argument("--window-size", type=int, default=2, help="Context window size")
    parser.add_argument("--learning_rate","-lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--embedding-size", type=int, default=15, help="Size of word embeddings")
    parser.add_argument("--hidden-layers", type=int, default=[20], help="Size of hidden layers")
    parser.add_argument("--hidden-activations", type=int, default=[RELU], help="Number of hidden layers")
    parser.add_argument("--xor_mode", type=bool, default=False, help="If you want to train a neural network for the xor task")
    parser.add_argument("--xavier", type=bool, default=True, help="If you want to initialize parameters with Xavier's method" )
    args = parser.parse_args()

    trainfile = args.train
    devfile = args.dev
    testfile = args.test
    window = args.window_size 

    possible_learning_rates = [0.001, 0.01, 0.1, 1, 10]
    

    #XOR Part
    """if args.xor_mode:
        # Creating training data
        X_train = [np.array(x) for x in [[0,1], [0,0], [1,0], [1,1]]]
        Y_train = [np.array(y) for y in [[1], [0],[1], [0]]]
        train_data = (X_train, Y_train)
        # Creating validation data
        X_dev = [np.array(x) for x in [[0,0], [0,1], [1,0], [1,1]]]
        Y_dev = [np.array(y) for y in [[0], [1],[1], [0]]]
        dev_data = (X_dev, Y_dev)
        # Creating test data
        X_test = [np.array(x) for x in [[0,0], [0,1], [1,1], [1,0]]]
        Y_test = [np.array(y) for y in [[0], [1],[0], [1]]] 
        test_data = (X_test, Y_test)
        for activ_funct in possible_hidden_activations:
            print(activ_funct)
            xor = nn.NeuralNetwork(args.hidden_layers, activ_funct, [0,1], 1, [0,1], 2, args.learning_rate , xor_mode=True)
            xor.train(train_data, dev_data, args.epochs)
            print(xor.evaluate(test_data))"""




    # TAGGER Part
    #Creating training data
    print("Lecture du corpus d'entraînement")
    train_sentences, train_vocabulary = rc.read_conllu(trainfile)
    classes = rc.get_classes(train_sentences)
    classes2index = {c: i for i, c in enumerate(classes)}
    train_data = rc.create_features(train_sentences, train_vocabulary, window, classes2index)
    
    # Creating test data
    test_sentences, test_vocabulary = rc.read_conllu(testfile, train_vocabulary)
    test_data = rc.create_features(test_sentences, train_vocabulary, window, classes2index)

    # Creating dev data
    dev_sentences, dev_vocabulary = rc.read_conllu(devfile, train_vocabulary)
    dev_data = rc.create_features(dev_sentences, train_vocabulary, window, classes2index)
    
    tagger = nn.NeuralNetwork(args.hidden_layers, args.hidden_activations, train_vocabulary, args.window_size, classes, args.embedding_size, args.learning_rate, args.xavier) 
    tagger.train(train_data, dev_data, args.epochs)
    tagger.evaluate(test_data)