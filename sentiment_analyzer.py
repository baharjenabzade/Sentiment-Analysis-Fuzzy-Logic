
import sklearn

import skfuzzy as fuzz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re

import tkinter as tk
from termcolor import colored


import time

columns = ["target", "id", "date", "flag", "user", "text"]
train_data = pd.read_csv("tweets_data.csv" , names=columns, encoding="ISO-8859-1")


def analyze_data():
    neg_tweets = train_data.target.value_counts()[0]
    pos_tweets = train_data.target.value_counts()[4]
    print(neg_tweets, pos_tweets)
    try:
        neu_tweets = train_data.target.value_counts()[2]
        print(neg_tweets)
    except:
        print("NO NEUTRAL TWEET FOUND!")



tweets = train_data.text
nltk.download("stopwords")
def stop_word_removal(tweet):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in tokens if not w.lower() in stop_words]

    final_tweet = " ".join(filtered_sentence)
    return final_tweet




def pre_process_tweet(tweet):
    final_tweet = stop_word_removal(tweet)
    return final_tweet


train_data["text"] = train_data.text.apply(pre_process_tweet)
print(train_data)
