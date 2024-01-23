!pip install tweepy

import tweepy

# Authentication
#consumerKey = ""
#consumerSecret = ""
#accessToken = ""
#accessTokenSecret = ""
#auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
#auth.set_access_token(accessToken, accessTokenSecret)
#api = tweepy.API(auth)

# Open/create a file to append data to
#csvFile = open('BatangKali.csv', 'a')

#Use csv writer
#import csv
#csvWriter = csv.writer(csvFile)

#queryTopic='Batang Kali OR batang Kali landslide OR #BatangKali OR Batang Kali,Selangor -filter:retweets'
#queryTopic='#BatangKaliLandslide OR #BatangKali OR #landslidedisaster OR #tanahruntuh OR Batang Kali,Selangor OR Batang Kali Landslide -filter:retweets'
#queryTopic='Batang Kali OR landslide OR updates OR Selangor OR victims -filter:retweets'
#queryTopic='(BatangKali) OR (Batang Kali landslide) OR (Batang Kali,Selangor) since:2022-12-1 -filter:retweets '
#queryTopic='(BatangKali) OR (Batang Kali Landslide) OR (#Malaysia#BatangKaliLandslide) -filter:retweets '

#tweets = tweepy.Cursor(api.search, q=queryTopic, lang='en' ).items(100)

#tweet_list=[]
#for tweet in tweets:
  #print(tweet.text)
  #tweet_list.append(tweet.text)

  # Write a row to the CSV file. I use encode UTF-8
  #csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
  #print(tweet.created_at, tweet.text)

#csvFile.close()
#len(tweet_list)

#tweet_list

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

df = pd.read_excel('BatangKaliTweets.xlsx')
df.head(2)

# adding header
#headerList = ['date/time','tweet']

# converting data frame to csv
#df.to_csv("BatangKali.csv", header=headerList, index=False)
#df.head(2)

#df=df.drop(['date/time'], axis=1)

import nltk
nltk.download('stopwords')

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')

def text_prep(x):
     corp = str(x).lower()
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]

     return lemmatize

import nltk
nltk.download('punkt')

import nltk
nltk.download('wordnet')

import nltk
nltk.download('omw-1.4')

preprocess_tag = [text_prep(i) for i in df['tweet']]
df["preprocess_tweet"] = preprocess_tag
df.head(3)

df['len'] = df['preprocess_tweet'].map(lambda x: len(x))

file = open('negative.txt', 'r')
neg_words = file.read().split()

file = open('positive.txt', 'r')
pos_words = file.read().split()

#file = open('booster_inc.txt', 'r')
#booster_inc = file.read().split()

#file = open('booster_decr.txt', 'r')
#booster_decr = file.read().split()

num_pos = df['preprocess_tweet'].map(lambda x: len([i for i in x if i in pos_words]))
df['pos_count'] = num_pos

num_neg = df['preprocess_tweet'].map(lambda x: len([i for i in x if i in neg_words]))
df['neg_count'] = num_neg

#num_booster_inc = df['preprocess_tweet'].map(lambda x: len([i for i in x if i in booster_inc]))
#df['booster_inc_count'] = num_booster_inc

#num_booster_dec = df['preprocess_tweet'].map(lambda x: len([i for i in x if i in booster_decr]))
#df['booster_dec_count'] = num_booster_dec

df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['len'], 2)
df

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

df['label'] = df['sentiment'].apply(sentiment)
df

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(5,5))
sns.countplot(x='label', data = df)

fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = df['label'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')

# Authentication
consumerKey = "Yn7McJf9v1dKST7BtVFT5lg8k"
consumerSecret = "dXCcfdgTnxWZfherQ5bz9fla6Z7kz0lJg1uXQrcHrscKytRTVT"
accessToken = "1385606469173661702-LB8uouus2UreliWDv6g9cRymAEBfkp"
accessTokenSecret = "i3Wi3leyTPjsKd8qzV2HOim97XcugFiUjNvOLLXOneeUc"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

# WOEID of London
woeid = 44418

# fetching the trends
trends = api.get_place_trends(id = woeid)

# printing the information
print("The top trends for the location are :")

for value in trends:
    for trend in value['trends']:
        print(trend['name'])
