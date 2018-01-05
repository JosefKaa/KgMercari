#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".

"""
#%%============================================================================
import pandas as pd
import numpy as np

#%%============================================================================
train = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/train.tsv')
train = train.iloc[0:15000,]  ## Short version of train, used for testing the code
train_X = train.drop(['train_id','price','name','brand_name','item_description','category_name'], 1)
train_Y = train[['price']]
train_n = np.shape(train)[0]
del train

train_cat = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_cat_fac.csv')
train_cat = train_cat.drop(train_cat.columns[0], axis=1)
train_brand = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_brand_fac.csv')
train_brand = train_brand.drop(train_brand.columns[0], axis=1)
train_name = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_prep_name.csv')
train_name = train_name.drop(train_name.columns[0], axis=1)
train_itdes = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_prep_itdes.csv')
train_itdes = train_itdes.drop(train_itdes.columns[0], axis=1)

#%%============================================================================
test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')
test = test.iloc[0:5000,]  ## Short version of test, used for testing the code
test_n = np.shape(test)[0]

test_cat = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_cat_fac.csv')
test_brand = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/test_brand_fac.csv')
test_name = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/test_prep_name.csv')
test_itdes = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/test_prep_itdes.csv')
# train.head()
#%%============================================================================
train_X = pd.concat([train_X, train_cat, train_brand, train_name, train_itdes], 1)

train_X.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_X_prep.csv', sep='\t')
train_Y.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_Y_prep.csv', sep='\t')
