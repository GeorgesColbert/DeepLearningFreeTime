# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



#import s3fs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#%matplotlib inline
sns.set()

#df = pd.read_csv('MD_BK_CrossJoin2.csv')



import sys
print(sys.path)

import os
#print(os.path.abspath(__file__))

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

scale = preprocessing.StandardScaler().fit(x_train)
X= scale.transform(x_train)

scale = preprocessing.StandardScaler().fit(x_test)
x = scale.transform(x_test)

# Specify the model
n_cols = x_train.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

earlystop = EarlyStopping(patience=50)

model.fit(X,y_train,validation_data = (x, y_test),nb_epoch=500,callbacks = [earlystop])


model.evaluate(x,y_test)

#import xgboost as xgb
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

rfy = RandomForestRegressor()


rfy.fit(x_train,y_train)

rfy_pred=rfy.predict(x_test)

mean_squared_error(y_test, rfy_pred)


#x = clf.predict_proba(X_test)
























 