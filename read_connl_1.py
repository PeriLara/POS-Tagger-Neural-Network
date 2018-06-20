#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Sortir les informations du connl dans un nouveau fichier .txt

import argparse
from collections import Counter, defaultdict
import numpy as np 

UNKNOWN= "__UNKNOWN__"
START= "START"
END="END"

TRAIN = 'train'
TEST = 'test'

def read_conllu(filename, voc=None): 
    """ lit un fichier au format conllu
        renvoie une liste de tuples mots de la phrase/cat des mots de la phrase 
        renvoie un le vocabulaire
            pour tous les mots ayant une seule occurence, 
            considérés comme UNKNOWN
            Et ajoute les mots en début et fin de phrase(START et END)

        si voc = None --> mode train, sinon mode test ou dev
    """
    try:
        sentences = []
        words, cats = [], []
        word2occ = Counter()
        for line in open(filename, "r", encoding="utf-8"):
            if not line.startswith("#") and line != "\n":
                line = line.lower().split("\t")
                word = line[1]
                words.append(word) #WORD
                cats.append(line[3]) #UPOSTAG
                word2occ[word] +=1

            elif line == "\n": #nouvelle phrase
                sentences.append((words, cats))
                words, cats = [], []
        
        if not voc:
            voc = [word for word, occ in word2occ.items() if occ > 1]
            voc.append(UNKNOWN)
            voc.append(START)
            voc.append(END)

        for words, cats in sentences:
            for i, w in enumerate(words):
                if w not in voc:
                    words[i] = UNKNOWN

        return sentences, voc

    except IOError:
        print("Problème lors de la lecture du fichier")
        exit()

def get_classes(sentences):
    classes = set()
    for _ ,cats in sentences:
        for c in cats:
            classes.add(c)
    return classes

def create_features(sentences, vocabulary, window, classe2index):
    X = []
    y = []
    for sentence in sentences:
        longueur_phrase = len(sentence[0])
        words, tags = sentence
        for i, (word, tag) in enumerate(zip(words, tags)):
            ex = []
            for w in range(-window, window + 1):
                if i + w < 0:
                    ex.append(START)
                elif i + w >= longueur_phrase:
                    ex.append(END)
                else:
                    context_word = words[i + w]
                    ex.append(context_word if context_word in vocabulary else UNKNOWN)


            X.append(ex)
            one_hot = np.zeros(shape=(len(classe2index),))
            one_hot[classe2index[tag]] = 1.0
            y.append(one_hot)
    return X, y