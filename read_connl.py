#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Sortir les informations du connl dans un nouveau fichier .txt

import argparse
from collections import Counter, defaultdict
import numpy as np 

UNKNOWN= "UNKNOWN"
TRAIN = 'train'
TEST = 'test'

def read_conllu(filename, window_size, voc=None): 
    """ lit un fichier au format conllu
        renvoie une liste de tuples mots de la phrase/cat des mots de la phrase 
        renvoie un le vocabulaire, mot:indice
            pour tous les mots ayant une seule occurence, 
            considérés comme UNKNOWN

        si voc = None --> mode train, sinon mode test ou dev
    """
    if not voc:
        train = "d'entrainement"
        print(f"Lecture du fichier {train}")
    else:
        test = "de test"
        print(f"Lecture du fichier {test}\n")
    try:
        sentences = []
        words, cats = [], []
        word2occ = Counter()
        for line in open(filename, "r",encoding="utf-8"):
            if line.startswith("#") == False and line != "\n":
                line = line.lower().split("\t")
                if line[3].upper() != "PUNCT" and line[2] != "_": # on ne prend pas en compte la ponctuation
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
            for i in range(-window_size, window_size+1):
                if i != 0:
                    voc.append(f"d{i}")
                    voc.append(f"f{i}")

        voc2index = {x:i for i,x in enumerate(voc)}

        return sentences, voc2index, voc

    except IOError:
        print("Problème lors de la lecture du fichier")
        exit()

def get_classes(sentences):
    classes = set()
    for _ ,cats in sentences:
        for c in cats:
            classes.add(c)
    return classes

def create_onehots(vocabulary, classes):
    """ Etape 1 :
            Crée des vecteurs creux pour les classes et le vocabulaire donnés
            retourne les vecteurs creux
    """
    nbr_words = len(vocabulary)
    X_onehots = np.zeros(shape=(nbr_words, nbr_words))

    nbr_classes = len(classes)
    Y_onehots = np.zeros(shape=(nbr_classes, nbr_classes))

    for i,_ in enumerate(vocabulary):
        X_onehots[i][i] = 1
    for j,_ in enumerate(classes):
        Y_onehots[j][j] = 1

    return X_onehots, Y_onehots

def create_features(sentences, X_onehots, vocabulary, Y_onehots, window, voc2index, classe2index):
    X = []
    y = []
    for sentence in sentences:
        longueur_phrase = len(sentence[0])
        for i, word_tag in enumerate(zip(sentence[0], sentence[1])):
            word, tag = word_tag
            x = np.zeros(shape=((window*2+1), len(vocabulary)))
            j = 0
            if word in vocabulary:
                index_word = voc2index[word]
            else:
                index_word = voc2index[UNKNOWN]
            for w in range(-window+1, window):
                if i + w < 0:   x[j][voc2index[f"d{w}"]] = 1
                elif i + w >= longueur_phrase:  x[j][voc2index[f"f{w}"]] = 1
                else:   x[j][index_word] = 1
                j+=1
            X.append(x)
            y.append(Y_onehots[classe2index[tag]])
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Corpus au format sgm")
    # lecture des arguments et des options passés en ligne de commande
    parser.add_argument('input', type = str, help = "le fichier connlu")
    parser.add_argument('window', type = int, help = "la fenêtre de features")
    #parser.add_argument('output', type = str, help = "le nom du fichier de sortie")
    #parser.add_argument('--classes','-c', type = bool, default=False, help = "si on veut sortir les POStags dans un fichier")

    args = parser.parse_args()

    sentences, voc2index, vocabulary = read_conllu(args.input, args.window)
    classes = get_classes(sentences)

    X_onehots, Y_onehots = create_onehots(vocabulary, classes)

    #Xy = create_features(sentences, X_onehots, vocabulary, Y_onehots, args.window, voc2index, classe2index)
    





"""
def get_features(sentence, i, window=3) :
    #Lit la phrase et retourne une fenetre de mots pour le mot à la place i
    longueur_phrase = len(sentence[0])
    assert (i < longueur_phrase), f" La phrase est trop courte pour i = {i} "
    features = []
    
    for w in range(-window+1, window):
        if i + w < 0:   features.append()
        elif i + w >= longueur_phrase:  features.append(f"f{w}")
        else:   features.append(sentence[0][i+w])
    
    assert(len(features) == (window*2)-1)
    
    return features

def generate_context_word_pairs(sentences, window_size, vocabulary):
    context_length = window_size*2
    for sentence, tags in sentences:
        sentence_length = len(sentence)
        for index, word in enumerate(sentence):
            context_words = []
            label_word   = []            
            

def write_txt(filename_input, filename_output):
    écrit dans un nouveau fichier filename_output, 
    1 ligne = mot \t cat
    écrit dans un nouveau fichier classes.txt, si classes = True
    1 ligne = cat (dédoublonné)
    print("Écriture du fichier \n")
    phrase_word2cat = read_conllu(filename_input)
    with open(filename_output,"w",encoding="utf-8") as fichier:
        for words, cats in phrase_word2cat:
            mot = str(word + "\t" + cat + "\n")
            fichier.write(mot)

    print(f"Fichier {filename_output} est écrit")
""" 


