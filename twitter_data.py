# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:35:20 2020

@author: Bhageerath
"""

import tweepy
import json
import sys
import pandas as pd

auth = tweepy.OAuthHandler("4aIqhS4aZrqnXqVpNSYUNgAYO", "2rs63y2jKBvwKFXjgjcIeXRumdbbcL4WZS3g96cvhV4FCfI53F")
auth.set_access_token("2998967156-lo6Uhg1w3fzQX9LG79KlllW1e1hePiX4eTLMkcH", "P25t7Mpbej20f9HgLgfHg6JKvntNBlBzCMgtXjR75WLIQ")
api = tweepy.API(auth)


twitter_data = []
df = pd.DataFrame()

class StdOutListener(tweepy.StreamListener):

    tweet_number=0   # class variable

    def __init__(self,max_tweets):
        self.max_tweets=max_tweets # max number of tweets

    def on_data(self, data):
        self.tweet_number+=1   
        try:
            tweet = json.loads(data)
            with open('your_data2.json', 'a') as my_file:
                json.dump(tweet, my_file)
        except BaseException:
            print('Error')
            pass
        if self.tweet_number>=self.max_tweets:
            sys.exit('Limit of '+str(self.max_tweets)+' tweets reached.')

    def on_error(self, status):
        print ("Error " + str(status))
        if status == 420:
            print("Rate Limited")
            return False
        
stream_listener = StdOutListener(10)
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
retail_tech_keywords = ["retailtechnology", "RetailTechnology"]
location_usa = [-171.791110603, 18.91619, -66.96466, 71.3577635769]
print(twitter_data)
stream.filter(track=retail_tech_keywords, locations=location_usa)