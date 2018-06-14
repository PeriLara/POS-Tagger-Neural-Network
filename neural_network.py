
import numpy as np
import math
import random
from time import time
from collections import defaultdict
import layers as l


RELU = "relu"
SIGMOID = "sigmoid"
SOFTMAX = "softmax"
TANH = "tanh"
LIN = "linear"

# ---------------------- Neural Network ---------------------

class NeuralNetwork():

    def __init__(self, layers, activ, vocabulary, window, classes, embedding=10, learning_rate=0.01):  
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
        self.embedding = l.Embedding_Layer(bfr_W, bfr_b, self.window) 
        # Initialization of the Lookup Layer
        bfr_W = np.random.uniform(((-(math.sqrt(6) / math.sqrt((layers[0]*(self.window*2+1)) + layers[1])))), (math.sqrt(6) / math.sqrt(layers[0]*(self.window*2+1)) + layers[1]), (layers[0]*(self.window*2+1), layers[1]))   
        bfr_b = np.random.uniform((-(math.sqrt(6) / math.sqrt(layers[1]))), (math.sqrt(6) / math.sqrt(layers[1])), ( layers[1], 1))
        self.param.append(l.Lookup_Layer(bfr_W, bfr_b, self.vocabulary, self.embedding))
        for i in range(len(activ)):
            mini = -(math.sqrt(6) / (math.sqrt(layers[i+1] + layers[i+2])))
            maxi = math.sqrt(6) / math.sqrt(layers[i+1] + layers[i+2])
            bfr_W = np.random.uniform(mini, maxi, (layers[i+1], layers[i+2]))
            bfr_b = np.random.uniform((-(math.sqrt(6) / math.sqrt(layers[i+2]))),(math.sqrt(6) / math.sqrt(layers[i+2])), (layers[i+2], 1) )
            if activ[i] == SOFTMAX:
                print(SOFTMAX, bfr_W.shape, bfr_b.shape)
                self.param.append(l.Softmax_Layer(bfr_W, bfr_b))
            elif activ[i] == RELU:
                print(RELU, bfr_W.shape, bfr_b.shape)
                self.param.append(l.RelU_Layer(bfr_W, bfr_b))
            elif activ[i] == LIN:
                self.param.append(l.Linear_Layer(bfr_W, bfr_b))
            else: # default
                self.param.append(l.Linear_Layer(bfr_W, bfr_b))

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
        assert res.shape == (1, (self.window*2+1)*self.layers[0])
        # Forward propagation for other layers which are fully connected
        for p in self.param: 
            res = p.forward_function(res)

        if pred:
            self.embedding.update()
            self.param[0].update()


        return self.param[-1].res_forward # output of the forward prop

    def back_propagation(self, y):
        # Backprop for the output layer, different from the others
        Wgrad, bgrad, backward = self.param[-1].backward_function(y, self.param[-2])
        assert Wgrad.shape == self.param[-1].W.shape
        assert bgrad.shape == self.param[-1].b.T.shape
        gradients = [(Wgrad, bgrad)]


        # Backprop for hiddenlayers + lookuplayer
        for i in reversed(range(len(self.param) - 1)):
            param = self.param[i]
            Wgrad, bgrad, backward = param.backward_function(backward) # avant i-1
            assert bgrad.shape == param.b.T.shape
            assert Wgrad.shape == param.W.shape
            gradients.append((Wgrad, bgrad)) 

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

