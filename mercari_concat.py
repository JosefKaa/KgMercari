# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".
Concatenate train and test.
"""
#%%============================================================================
import pandas as pd
import numpy as np

#%%============================================================================
train = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/train.tsv')
test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')
# train.head()
# train = train.iloc[0:10,]  ## Short version of train, used for testing the code
# test = test.iloc[0:5000,]  ## Short version of test, used for testing the code
train_n = np.shape(train)[0]
test_n = np.shape(test)[0]

train_X = train.drop(['price'], 1)
# train_Y = train[['price']]

tot = pd.concat([train_X, test]) # concatenate train and test to preprocess
