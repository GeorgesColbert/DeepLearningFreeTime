#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:58:48 2020

@author: georgescolbert
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


import keras


from keras.datasets import imdb, reuters
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


# Import relevant classes/functions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


#Preprocessing text
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')

import gensim
max_features = 40000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#max length of documents in  training text
max_length=150

#max_length = max([len(s) for s in x_train])  
x_train = pad_sequences(x_train, maxlen=max_length)
x_test= pad_sequences(x_test, maxlen=max_length)

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)



from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, GRU, Dropout, Bidirectional
from keras.initializers import Constant
from keras.optimizers import SGD
#define model
model = Sequential()

model.add(Embedding(40000,178,input_length=max_length))
#model.add(Bidirectional(LSTM(74, return_sequences=False)))
model.add(GRU(units=52))

model.add(Dropout(rate=0.25))

model.add(Dense(1, activation='sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.5, clipnorm=1.0), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystop = EarlyStopping(patience=2)


print('Train...')
model.fit(x_train,y_train,batch_size=500,validation_data = (x_test,y_test),epochs=10,callbacks = [earlystop])

#Best Accuracy so far: 84%

model.evaluate(x_test,y_test)



from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

max_features = 40000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32


model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])




