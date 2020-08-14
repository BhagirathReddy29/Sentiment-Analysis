# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:55:31 2020

@author: Bhageerath
"""

import csv
import re
from nltk.corpus import stopwords
from string import punctuation 
import nltk
import pandas as pd
nltk.download('stopwords') 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def buildTrainingData(twitterDataFile):
    training_data = []
    with open(twitterDataFile, "rt") as f:
        lineReader = csv.reader(f,delimiter=',', quotechar="\"")
        for row in lineReader:
            training_data.append({"company":row[0], "label":row[1], "text":row[2]})
    return training_data

def buildTestingData(testcsvDataFile):
    testing_data = []
    with open(testcsvDataFile, "rt", encoding="utf-8") as f:
        lineReader = csv.reader(f,delimiter=',', quotechar="\"")
        for row in lineReader:
            if row != []:
                testing_data.append({"text":row[0], "label":"dummy"})
    return testing_data

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
            text, label = tuple(zip(*processedTweets))
        return pd.DataFrame({'text': text, 'label': label})
    
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        #tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return tweet

def toUnigramTfDf(train_data, test_data):
    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_vectorizer.fit(train_data)
    X_train_unigram = unigram_vectorizer.transform(train_data)
    Y_train_unigram = unigram_vectorizer.transform(test_data)
    unigram_tf_idf_transformer = TfidfTransformer()
    unigram_tf_idf_transformer.fit(X_train_unigram)
    X_train_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)
    Y_train_tf_idf = unigram_tf_idf_transformer.transform(Y_train_unigram)
    return (X_train_tf_idf, Y_train_tf_idf)

def train_and_show_scores(X: csr_matrix, y: np.array, title: str):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.95, stratify=y
    )
    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')
    return clf


training_data_raw = buildTrainingData("C:/Users/bvulu/Documents/AIT-664/full-corpus.csv")
training_data_raw = training_data_raw[1:-1]
tweetProcessor = PreProcessTweets()
training_data = tweetProcessor.processTweets(training_data_raw)
#training_features = toUnigramTfDf(training_data['text'].values)
training_labels = training_data['label'].values
testing_data_raw = buildTestingData("C:/Users/bvulu/Documents/AIT-664/test123457.csv")
testing_data = tweetProcessor.processTweets(testing_data_raw)
training_features,testing_features = toUnigramTfDf(training_data['text'].values,testing_data['text'].values)




clf = train_and_show_scores(training_features, training_labels, 'Unigram Counts')
scores = cross_val_score(clf, training_features, training_labels, cv=5)
scores1 = clf.predict(testing_features)

testing_data_str = " ".join(str(e) for e in testing_data['text'].values)
wordcloud = WordCloud().generate(testing_data_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")



k = pd.DataFrame(scores1)
print(k[0].value_counts())
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)