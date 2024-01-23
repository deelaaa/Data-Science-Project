import tweepy
import pandas as pd
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Authentication
consumerKey = ""
consumerSecret = ""
accessToken = ""
accessTokenSecret = ""
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)
#api = tweepy.API(auth, wait_on_rate_limit=True)

limit = 1000
date_since_pro = "202212170000"
#tweets = tweepy.Cursor(api.search, q="(#Batangkali) OR (Batang Kali landslide)", tweet_mode = 'extended', location="Malaysia", lang="en").items(limit)
tweets=tweepy.Cursor(api.search_full_archive,environment_name='FullArchive', query='#BatangKaliLandslide', fromDate=date_since_pro).items(limit)

columns = ['User', 'Tweets', 'Likes', 'Location', 'Language']
data = []

for tweet in tweets:
    #data.append([tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.user.location, tweet.lang])
    data.append([tweet.user.screen_name, tweet.text, tweet.favorite_count, tweet.user.location, tweet.lang])

df = pd.DataFrame(data, columns=columns)

new_df = df.copy()

header = ["User","Tweets","Likes","Location","Language"]
new_df.to_csv('generatedTweets.csv', columns = header)

#print(df)
len(new_df)

new_df

new_df = new_df[~new_df.Tweets.str.contains("RT")]
new_df = new_df.reset_index(drop=True)

new_df

new_df = new_df[new_df.Language.str.contains("en")]

new_df

##first cleaning
### Checking Missing values in the Data Set and printing the Percentage for Missing Values for Each Columns ###

count = new_df.isnull().sum().sort_values(ascending=False)
percentage = ((new_df.isnull().sum()/len(new_df)*100)).sort_values(ascending=False)
missing_data = pd.concat([count, percentage], axis=1,keys=['Count','Percentage'])

missing_data

new_df.dropna(axis=0, how='any', inplace=True)

new_df

header2 = ["User","Tweets","Likes","Location","Language"]
new_df.to_csv('precleaned.csv', columns = header2)

new_df = new_df[~new_df.Location.str.contains("TÃ¼rkiye")]
new_df = new_df[~new_df.Location.str.contains("Switzerland")]
new_df = new_df[~new_df.Location.str.contains("World")]
new_df = new_df[~new_df.Location.str.contains("Singapore")]
new_df = new_df[~new_df.Location.str.contains("dandelion")]
new_df = new_df[~new_df.Location.str.contains("Somewhere")]
new_df = new_df[~new_df.Location.str.contains("Arohacity")]
new_df = new_df[~new_df.Location.str.contains("she/her")]
new_df = new_df[~new_df.Location.str.contains("here")]
new_df = new_df[~new_df.Location.str.contains("Alberta")]
new_df = new_df.drop([79, 82, 83, 84, 85, 86, 87, 91, 92, 94, 95, 96, 101, 102, 103])

new_df

count_row = new_df.shape[0]
count_row

lemma = WordNetLemmatizer()

def data_processing(x):
    corp = str(x).lower()
    corp = re.sub(r"https\S+|www\S+https\S+", '',corp, flags=re.MULTILINE)
    corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
    corp = re.sub(r'\@w+|\#','',corp)
    corp = re.sub(r'[^\w\s]','',corp)
    Tweets_clean = corp

    return Tweets_clean

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

def data_processing2(x):
    corp = str(x).lower()
    corp = re.sub(r"https\S+|www\S+https\S+", '',corp, flags=re.MULTILINE)
    corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
    corp = re.sub(r'\@w+|\#','',corp)
    corp = re.sub(r'[^\w\s]','',corp)
    Tweets_tokens = word_tokenize(corp)
    filtered_Tweets = [t for t in Tweets_tokens if t not in stop_words]
    lemmatize = [lemma.lemmatize(w) for w in filtered_Tweets]

    return lemmatize

new_df['cleaned_Tweets'] = pd.DataFrame(new_df['Tweets'].apply(data_processing))

preprocess_tag = [data_processing2(i) for i in new_df['Tweets']]
new_df["preprocess_Tweets"] = preprocess_tag

new_df = new_df.drop_duplicates('Tweets')

new_df

new_df['total_len'] = new_df['preprocess_Tweets'].map(lambda x: len(x))

new_df

file = open('negative.txt', 'r')
neg_words = file.read().split()
file = open('positive.txt', 'r')
pos_words = file.read().split()
file = open('booster_inc.txt', 'r')
boosterinc_words = file.read().split()
file = open('booster_decr.txt', 'r')
boosterdecr_words = file.read().split()
file = open('negation.txt', 'r')
negation_words = file.read().split()
file = open('negation.txt', 'r')

num_pos = new_df['preprocess_Tweets'].map(lambda x: len([i for i in x if i in pos_words]))
new_df['pos_count'] = num_pos
num_neg = new_df['preprocess_Tweets'].map(lambda x: len([i for i in x if i in neg_words]))
new_df['neg_count'] = num_neg
num_boosterinc = new_df['preprocess_Tweets'].map(lambda x: len([i for i in x if i in boosterinc_words]))
new_df['boosterinc_count'] = num_boosterinc
num_boosterdecr = new_df['preprocess_Tweets'].map(lambda x: len([i for i in x if i in boosterdecr_words]))
new_df['boosterdecr_count'] = num_boosterdecr
num_negation = new_df['preprocess_Tweets'].map(lambda x: len([i for i in x if i in negation_words]))
new_df['negation_count'] = num_negation

##sentiment score = polarity
new_df['polarity'] = round((new_df['pos_count'] - new_df['neg_count']) / new_df['total_len'], 2)

new_df

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

def summarization_score(label):
    if label <0:
        return "-1"
    elif label ==0:
        return "0"
    elif label>0:
        return "+1"

new_df['sentiment'] = new_df['polarity'].apply(sentiment)

new_df['score'] = new_df['polarity'].apply(summarization_score)

new_df

new_df.shape

new_df[['sentiment','score']].value_counts()

fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = new_df)

fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = new_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')

pos_tweets = new_df[new_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()

neg_tweets = new_df[new_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()

neutral_tweets = new_df[new_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()

text = ' '.join([word for word in pos_tweets['cleaned_Tweets']])
plt.figure(figsize=(15,10), facecolor='None')
wordcloud = WordCloud(max_words=500, width = 1600, height = 800).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews\n', fontsize = 19)
plt.show()

text = ' '.join([word for word in neg_tweets['cleaned_Tweets']])
plt.figure(figsize=(15,10), facecolor='None')
wordcloud = WordCloud(max_words=500, width = 1600, height = 800).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews\n', fontsize = 19)
plt.show()

text = ' '.join([word for word in neutral_tweets['cleaned_Tweets']])
plt.figure(figsize=(15,10), facecolor='None')
wordcloud = WordCloud(max_words=500, width = 1600, height = 800).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Most frequent words in neutral reviews\n', fontsize = 19)
plt.show()
