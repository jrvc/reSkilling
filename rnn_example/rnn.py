# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:08:18 2017

@author: Raul Vazquez
"""
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import utils
import os
import io


# ======================================================
# ================= PREPROCESS: ========================
# ======================================================

# set current working directory in the place where the data is
os.chdir('C:\\Users\\Raul Vazquez\\Desktop\\reSkilling\\reSkilling\\rnn_example')

# Download NLTK model data (you need to do this once)
nltk.download("book")

# set parameters
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# read file
file = io.open('reddit-comments-2015-08.csv','r',encoding='utf-8')
text = file.readlines()

# divide by sentences and include the tokens and begining/end of each sentence
sentences = itertools.chain(*[nltk.sent_tokenize(text[x].lower()) for x in range(1,len(text))])
sentences = ["SENTENCE_START %s SENTENCE_END" %line for line in sentences]

# Tokenize the sentences into words
toked_sentences = [nltk.word_tokenize(x) for x in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*toked_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

#print("Using a vocabulary size of %d." % vocabulary_size)
#print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i,sent in enumerate(toked_sentences):
    toked_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

#print("\nExample sentence: '%s'" % sentences[0])
#print("\nExample sentence after Pre-processing: '%s'" % toked_sentences[0])



# Create the training data
'''
   TRAINING DATA IN RNN FOR TEXT PREDICTION:
   A training example  x_train  may look like [0, 179, 341, 416],
   where 0 corresponds to SENTENCE_START. The corresponding label
   y_train  would be [179, 341, 416, 1]. 
   
   Remember that our goal is to predict the next word, so y_train
   is just the x_train vector shifted by one position with the last
   element being the SENTENCE_END token.
'''
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in toked_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in toked_sentences])


# ==========================================================================
# ================= USE THE GUYs RNN CLASS FOR THEANO: =====================
# ==========================================================================
from rnn_theano import RNNTheano, gradient_check_theano

np.random.seed(10)
# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 5
model = RNNTheano(grad_check_vocab_size, 10)
gradient_check_theano(model, [0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNTheano(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)

# LOAD the model parameters that he trained:
model = RNNTheano(vocabulary_size, hidden_dim=50)
utils.load_model_parameters_theano('./data/trained-model-theano.npz', model)

# TRAIN the model if wanted, but he said he trained his for 20hrs:
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)


# ======================================================
# ================= BUILD OWN RNN: =====================
# ======================================================
'''
    The input  x = X-train[i]  will be a sequence of words
    and each  x_t = x[t]  is a single word.
    
    Can't simply use a word index (like 36) as an input.
    Instead, we represent each word as a canonical vector
    of the canonical basis for R^{vocabulary_size}(like the 36th canonical vector)
    
    THE NN:
        s_t = tanh(U*x_t + W*s_{t-1})
        o_t = softmax(V*s_t)
    where:
        x_t,o_t \in R^C
        s_t     \in R^H
        U       \in R^{H x C}
        V       \in R^{C x H}
        U       \in R^{H x H}
    if assumed C as vocabulary size and H as hidden layer size
    
NOTE: because x_t is a canonical vector, multiplying it with U is essentially}
      the same as selecting a column of U, so we don't need to perform the full multiplication. 
      Then, the biggest matrix multiplication in our network is  V*s_t    
'''

# initialize with a random small number




# ======================================================
# ================= GENERATE TEXT: =====================
# ======================================================
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)






