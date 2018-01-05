# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".
Factorize (labelize) category name and brand and save it to csv.
"""
#%%============================================================================
import pandas as pd
import numpy as np

#%%============================================================================
train = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/train.tsv')
test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')
# train.head()
#train = train.iloc[0:15000,]  ## Short version of train, used for testing the code
#test = test.iloc[0:5000,]  ## Short version of test, used for testing the code
train_n = np.shape(train)[0]
test_n = np.shape(test)[0]

train_X = train.drop(['price'], 1)
# train_Y = train[['price']]

tot = pd.concat([train_X, test]) # concatenate train and test to preprocess

#%%============================================================================
#### Factorize the categories.
tot_cat = tot.category_name
tot_cat_split = tot_cat.str[:].str.split('/', expand=True)
tot_cat_split.columns = ['cat1','cat2','cat3','cat4','cat5']

for i in range(5):
    tot_cat_split.iloc[:,i] = pd.factorize(tot_cat_split.iloc[:,i])[0]
# Nan is "-1" after being facterized
 
train_cat_fac = tot_cat_split.iloc[0:train_n,]
test_cat_fac = tot_cat_split.iloc[train_n:(train_n+test_n),]

train_cat_fac.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep/train_cat_fac.csv', sep='\t')
test_cat_fac.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep/test_cat_fac.csv', sep='\t')

#%%============================================================================
#### Factorize the brands.
tot_brand = pd.factorize(tot.brand_name)[0]
# Nan is "-1" after being facterized
train_brand_fac = tot_brand[0:train_n,]
test_brand_fac = tot_brand[train_n:(train_n+test_n)]
train_brand_fac = pd.DataFrame(train_brand_fac)
test_brand_fac = pd.DataFrame(test_brand_fac)

train_brand_fac.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep/train_brand_fac.csv', sep='\t')
test_brand_fac.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep/test_brand_fac.csv', sep='\t')



























