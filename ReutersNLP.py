#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:19:33 2020

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
max_features = 30000
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
#max length of documents in  training text
max_length=200
tokenizer = Tokenizer(num_words=max_features)
#max_length = max([len(s) for s in x_train])  

#x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
#x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

x_train = pad_sequences(x_train, maxlen=max_length)
x_test= pad_sequences(x_test, maxlen=max_length)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

word_index = reuters.get_word_index(path="reuters_word_index.json")
num_words= len(word_index) + 1

#max_length = x_train.shape[1]

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, GRU, Dropout, Bidirectional
from keras.initializers import Constant
from keras.optimizers import SGD
#define model
model = Sequential()

model.add(Embedding(30000,200,input_length=max_length, trainable=True))
model.add(GRU(units=200,return_sequences=True))
model.add(Bidirectional(LSTM(150, return_sequences=False,dropout=0.2, recurrent_dropout=0.15)))


model.add(Dropout(rate=0.25))

model.add(Dense(46, activation='softmax'))

#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.5, clipnorm=1.0), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#earlystop = EarlyStopping(monitor='val_accuracy',mode='max',min_delta=1,patience=3)

earlystop = EarlyStopping(patience=3)

model.fit(x_train,y_train,batch_size=500,validation_data = (x_test,y_test),epochs=100,callbacks = [earlystop])

#Best accuracy: 56%





model.evaluate(x_test,y_test)






(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_features)

x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')


num_classes = max(y_train) + 1


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping
from sklearn import preprocessing


# Specify the model
n_cols = x_train.shape[1]
model = Sequential()
model.add(Dense(200, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))



model.add(Dense(46, activation='softmax'))
#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.5, clipnorm=1.0), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystop = EarlyStopping(patience=3)

model.fit(x_train,y_train,validation_data = (x_test, y_test),nb_epoch=100,callbacks = [earlystop])
































