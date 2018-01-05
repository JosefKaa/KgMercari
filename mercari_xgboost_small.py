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
train_X = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_X_prep.csv')
train_Y = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_Y_prep.csv')
#test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')

train_X.columns = [str(aa) for aa in range(28764)] # rename train_X to eliminate same column names
train_n = np.shape(train_X)[0]
#test_n = np.shape(test)[0]

#tot = pd.concat([train_X, test]) # concatenate train and test to preprocess

X_train, X_cv, Y_train, Y_cv = tts(train_X, train_Y, test_size=0.2)
#%%============================================================================

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
predictions = gbm.predict(X_cv)