#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:54:21 2020

@author: georgescolbert
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


import keras


from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from sklearn.datasets import fetch_20newsgroups

#Preprocessing text
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')

import gensim




cat = ['talk.politics.guns',
 'talk.politics.mideast',
 'soc.religion.christian']


ntrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers'),categories=cat)


ntest = fetch_20newsgroups(subset='test',remove=('headers', 'footers'),categories=cat)


#Transform data

review_lines = list()

lines = ntrain['data']

for line in lines:
    tokens = word_tokenize(line)
    
    #convert to lower case
    tokens = [w.lower() for w in tokens]
    
    #remove punctuation form each word
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    #remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
   
    #filter out stop words
    stop_words=set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)
  
    
review_tlines = list()

testlines = ntest['data']

for line in testlines:
    tokens = word_tokenize(line)
    
    #convert to lower case
    tokens = [w.lower() for w in tokens]
    
    #remove punctuation from each word
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    #remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
   
    #filter out stop words
    stop_words=set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_tlines.append(words)
      
################Declare embedding Dimension

EMBEDDING_DIM=500
  
genmodel = gensim.models.Word2Vec(sentences=review_lines,size=EMBEDDING_DIM,window=5,workers=10,min_count=3)
    
words= list(genmodel.wv.vocab)  

print('vocabulary size: %d' % len(words))

filename = 'news_embedding_word2vec.txt'

genmodel.wv.save_word2vec_format(filename,binary=False)

######################


import os

embeddings_index = {}
#Pull embedding if needed
f = open(os.path.join('','news_embedding_word2vec.txt'),encoding='utf-8') 
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

EMBEDDING_DIM=500  
 
######################
 
 # Import relevant classes/functions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

#review_lines = ntrain.data
#review_tlines = ntest.data


#max length of documents in  training text
#max_length = max([len(s) for s in review_lines])  
max_length= 200
    
# Create and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_lines)

# Prepare the data
prep_data = tokenizer.texts_to_sequences(review_lines)
prep_data = pad_sequences(prep_data, maxlen=max_length,padding='post')


test_data = tokenizer.texts_to_sequences(review_tlines)
test_data = pad_sequences(test_data, maxlen=max_length,padding='post')
# Prepare the labels
prep_labels = to_categorical(ntrain.target)
test_labels = to_categorical(ntest.target)


#map embeddings from the loaded word2vec model for each word to the tokenizer_obj.word_index vocabulary 

#create a matrix with of word vectors.
word_index = tokenizer.word_index
print('found %s unique tokens' % len(word_index))

num_words= len(word_index) + 1



###################
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM)) 

for world, i in word_index.items():
    if i>num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
    
        
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, GRU, Dropout, Bidirectional
from keras.initializers import Constant
from keras.optimizers import SGD
#define model
model = Sequential()

#embedding_layer = Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=max_length,trainable=False)
#model.add(embedding_layer)

model.add(Embedding(num_words,128,input_length=max_length,trainable=True))
#model.add(LSTM(74, return_sequences=True,activation='relu'))
model.add(Bidirectional(LSTM(74, return_sequences=False)))

#model.add(GRU(units=72,return_sequences=True,activation='relu',dropout=0.2,recurrent_dropout=0.15))

model.add(Dropout(rate=0.25))
#model.add(LSTM(44, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
model.add(Dense(3, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.5, clipvalue=4.0), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystop = EarlyStopping(patience=5)

model.fit(prep_data,prep_labels,batch_size=150,validation_data = (test_data, test_labels),epochs=20,callbacks = [earlystop])


model.evaluate(test_data, test_labels)
#Best Accuracy so far: 84%

    
    
    