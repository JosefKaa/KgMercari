# -*- coding: utf-8 -*-
"""
For the Kaggle competition, "Mercari Price Suggestion Challenge".
Use bag of words to preprocess texts in "names" and "item descriptions", and save it to csv.
"""

#%%============================================================================
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

#%%============================================================================
train = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/train.tsv')
test = pd.read_table('C:/Users/Xylthx/Desktop/Semester 10/Kaggle/test.tsv')
# train.head()
# train = train.iloc[0:1000,]  ## Short version of train, used for testing the code
# test = test.iloc[0:500,]  ## Short version of test, used for testing the code
train_n = np.shape(train)[0]
test_n = np.shape(test)[0]

train_X = train.drop(['price'], 1)
# train_Y = train[['price']]

tot = pd.concat([train_X, test]) # concatenate train and test to preprocess

#%%============================================================================
def bow_prep(train_names, num_of_features):
    #### Preprocess names and descriptions using bag of words
    train_names = pd.Series.to_frame(train_names)
    train_names['word_tokens'] = '[123]'
    train_names['filtered_tokens'] = '[123]'

    all_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    lemma = nltk.wordnet.WordNetLemmatizer()
    
    # Tokenizing the reviewText and removing the stop words from training dataset
    for i in range(train_names.shape[0]):
      train_names.iloc[i,1] = tokenizer.tokenize(train_names.iloc[i,0])  
      word_tokens = [w.lower() for w in train_names.iloc[i, 1]]
      filtered_tokens = [lemma.lemmatize(w) for w in word_tokens if w not in sw.words("english")]
      train_names.iloc[i, 2] = filtered_tokens
      all_words += filtered_tokens
      if i%1000 == 0:
        print(i)
    
    # Creating the feature space of top 50 words for each token for both test and train dataset
    freq_words = nltk.FreqDist(all_words)
    freq_words = pd.DataFrame(freq_words.most_common(num_of_features))
    word_features = list(freq_words[0])
    
    def find_features(document):  
      '''
      To create the feature space from the given tokens for each review
      Input: Filtered_tokens: for each reviewText
      Output: features: the feature space of given review based on the top 3000 most 
      common words
      '''
      words = set(document)
      features = {}
      for w in word_features:
          features[w] = (w in words)
      return features
    
    featuresets = [find_features(tokens) for tokens in train_names.filtered_tokens]
    train_names_fnl = pd.DataFrame(featuresets)
    return(train_names_fnl)

#%%============================================================================
# BoW for "name"
tot_names_fnl = bow_prep(tot.name, 1000)
train_names_fnl = tot_names_fnl.iloc[0:train_n,]
test_names_fnl = tot_names_fnl.iloc[train_n:(train_n+test_n),]

train_names_fnl.to_csv('C:/Users/Xylthx/Desktop/Semester 9.5/Kaggle/prep/train_name_fnl.csv', sep='\t')
test_names_fnl.to_csv('C:/Users/Xylthx/Desktop/Semester 9.5/Kaggle/prep/test_name_fnl.csv', sep='\t')

#%%============================================================================
# BoW for "item_description"
tot_desc_fnl = bow_prep(tot.item_description, 5000)
train_desc_fnl = tot_desc_fnl.iloc[0:train_n,]
test_desc_fnl = tot_desc_fnl.iloc[train_n:(train_n+test_n),]

train_desc_fnl.to_csv('C:/Users/Xylthx/Desktop/Semester 9.5/Kaggle/prep/train_desc_fnl.csv', sep='\t')
test_desc_fnl.to_csv('C:/Users/Xylthx/Desktop/Semester 9.5/Kaggle/prep/test_desc_fnl.csv', sep='\t')
