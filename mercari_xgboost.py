#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".

"""
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split as tts

#%%============================================================================
train = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/train.tsv')
test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')
# train.head()

train = train.iloc[0:10000,]  ## Short version of train, used for testing the code
test = test.iloc[0:5000,]  ## Short version of test, used for testing the code

train_n = np.shape(train)[0]
test_n = np.shape(test)[0]

train_X = train.drop(['price'], 1)
train_Y = train[['price']]

#tot = pd.concat([train_X, test]) # concatenate train and test to preprocess

X_train, X_cv, Y_train, Y_cv = tts(train_X, train_Y, test_size=0.2)
#%%============================================================================

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
predictions = gbm.predict(X_cv)