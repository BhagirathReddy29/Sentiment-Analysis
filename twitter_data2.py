# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:38:47 2020

@author: Bhageerath
"""

import tweepy
from sqlalchemy.exc import ProgrammingError
import json
import csv
import sys
import GetOldTweets3 as got
import re
from nltk.corpus import stopwords
from string import punctuation 
import nltk
import pandas as pd
nltk.download('stopwords') 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from datetime import date
from dateutil.relativedelta import relativedelta

os.path.join(os.path.dirname(__file__))

MAXTWEETS = 1000
FULLCORPUSTRAINDATA = "full-corpus.csv"
LIVESTREAMCSVFILENAME = "live_stream.csv"
OLDDATACSVFILENAME = "old_data.csv"
TRACK_TERMS = ["retail technology", "retailtechnology", "RetailTechnology", "Retail Technology"]
CSVCOLUMNS = ["text"]
LOCATION = [-171.791110603, 18.91619, -66.96466, 71.3577635769]

#To get live stream tweets you need to use your twitter API keys which you get from twitter developer app
TWITTER_APP_KEY = "ENTER YOUR TWITTER_APP_KEY"
TWITTER_APP_SECRET = "ENTER YOUR TWITTER_APP_SECRET"
TWITTER_KEY= "ENTER YOUR TWITTER_ACCESS_KEY"
TWITTER_SECRET= "ENTER YOUR TWITTER_ACCESS_SECRET_KEY"

TODAYS_DATE = str(date.today())
THREE_MONTHS_BACK = str(date.today() + relativedelta(months=-3))
UNITED_STATES_NAME = "United States Of America"
RETAIL_TECHNOLOGY_KEY = "retailtechnology"

class StreamListener(tweepy.StreamListener):
    tweet_number = 0
    
    def __init__(self, no_of_tweets):
        self.no_of_tweets = no_of_tweets

    def on_data(self, status):
        tweet = json.loads(status)
        try:
            self.tweet_number+=1
            with open(LIVESTREAMCSVFILENAME, 'a', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, CSVCOLUMNS)
                writer.writerow(dict(
                    text=tweet['text'],
                ))
        except ProgrammingError as err:
            print(err)
        if self.tweet_number>=self.no_of_tweets:
            sys.exit('Limit of '+str(self.no_of_tweets)+' tweets reached.')
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

auth = tweepy.OAuthHandler(TWITTER_APP_KEY, TWITTER_APP_SECRET)
auth.set_access_token(TWITTER_KEY, TWITTER_SECRET)
api = tweepy.API(auth)

stream_listener = StreamListener(MAXTWEETS)
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
print("Strating a new thread to stream live tweets")
stream.filter(track=TRACK_TERMS, locations=LOCATION, is_async=True)

#Old tweets data
print("Collects old tweets data")
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(RETAIL_TECHNOLOGY_KEY).setSince(THREE_MONTHS_BACK).setUntil(TODAYS_DATE).setNear(', ' + UNITED_STATES_NAME).setMaxTweets(MAXTWEETS)
for tweet in got.manager.TweetManager.getTweets(tweetCriteria):
    with open(OLDDATACSVFILENAME, 'a', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, CSVCOLUMNS)
        writer.writerow(dict(
                text=tweet.text
            ))

##Sentiment Analysis
def buildTrainingData(twitterDataFile):
    training_data = []
    with open(twitterDataFile, "rt") as f:
        lineReader = csv.reader(f,delimiter=',', quotechar="\"")
        for row in lineReader:
            training_data.append({"company":row[0], "label":row[1], "text":row[2]})
    return training_data

def buildTestingData(testcsvDataFiles):
    testing_data = []
    for testcsvDataFile in testcsvDataFiles:
        with open(testcsvDataFile, "rt", encoding="utf-8") as f:
            lineReader = csv.reader(f,delimiter=',', quotechar="\"")
            for row in lineReader:
                if row != []:
                    testing_data.append({"text":row[0], "label":"dummy"})
    return testing_data

#Preprocess tweets to clear of uneccesary symbols that cannot be prased properly.
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

#Convert text data into numerical data freuquency counts of some commonly used stop words.
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

#Train the model
def train(X: csr_matrix, y: np.array, title: str):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.95, stratify=y
    )
    #Neural net with tanh activation and two hidden layers with adam optimizer hyperparameters found through
    #cross validation.
    clf = MLPClassifier(solver='adam', activation='tanh', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    clf.fit(X_train, y_train)
    return clf


print("Starting Symantic Analysis")
training_data_raw = buildTrainingData(FULLCORPUSTRAINDATA)
training_data_raw = training_data_raw[1:-1]
tweetProcessor = PreProcessTweets()
training_data = tweetProcessor.processTweets(training_data_raw)
training_labels = training_data['label'].values
testing_data_raw = buildTestingData([LIVESTREAMCSVFILENAME, OLDDATACSVFILENAME])
testing_data = tweetProcessor.processTweets(testing_data_raw)
training_features,testing_features = toUnigramTfDf(training_data['text'].values,testing_data['text'].values)




clf = train(training_features, training_labels, 'Unigram Counts')
scores = cross_val_score(clf, training_features, training_labels, cv=5)
scores1 = clf.predict(testing_features)
k = pd.DataFrame(scores1)
print(k[0].value_counts())
#Relatively lowers accuracy is due to fact that the dataset containing unequal number of types of tweets for example
#there are more neutral tweets than positive + negative combined.
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

#WordCloud for twitter data
print("Generating word cloud")
testing_data_str = " ".join(str(e) for e in testing_data['text'].values)
wordcloud = WordCloud().generate(testing_data_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


