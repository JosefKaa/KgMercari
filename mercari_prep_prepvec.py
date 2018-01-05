#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".
To preprocess and vectorize text features "name" and "item description".
Remove stopwords, lemmatize, and use tf-idf to vectorize.
Save as csv.
"""
#==============================================================================
# import ipyparallel as ipp
# from joblib import Parallel, delayed
# import multiprocessing
#==============================================================================
import string
import pandas as pd
import numpy as np

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

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
# code adapted from "https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html"

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

def lemmatize(token, tag):
    tag = {
		'N': wn.NOUN,
         'V': wn.VERB,
         'R': wn.ADV,
         'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    
    return WordNetLemmatizer().lemmatize(token, tag)

def tokenize(document):
    for sent in sent_tokenize(document):
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')
			
            stopwords = set(sw.words('english'))
            if token in stopwords:
                continue
            
            punct = set(string.punctuation)
            if all(char in punct for char in token):
                continue
				
            lemma = lemmatize(token, tag)
            yield lemma


def nltkPreprocessing(X):
    return [list(tokenize(doc)) for doc in X]
    
def prepandvec(X):
    # nltk preprocess
    X = nltkPreprocessing(X)    
    # vectorize
    vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(X)
    #feature_names = vectorizer.get_feature_names()
    prepX = tfidf_matrix.todense()
    return prepX

#%%============================================================================
prep_name = prepandvec(tot.name)
train_prep_name = pd.DataFrame(prep_name[0:train_n,])
test_prep_name = pd.DataFrame(prep_name[train_n:(train_n+test_n)])

train_prep_name.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_prep_name.csv', sep='\t')
test_prep_name.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/test_prep_name.csv', sep='\t')

#%%============================================================================
prep_itdes = prepandvec(tot.item_description)
train_prep_itdes = pd.DataFrame(prep_itdes[0:train_n,])
test_prep_itdes = pd.DataFrame(prep_itdes[train_n:(train_n+test_n)])

train_prep_itdes.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/train_prep_itdes.csv', sep='\t')
test_prep_itdes.to_csv('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/prep small/test_prep_itdes.csv', sep='\t')
#==============================================================================
# def main():
#     X = tot.name
#     vectorize(X)
#        
# if __name__ == "__main__":
#     main()
#==============================================================================
    
