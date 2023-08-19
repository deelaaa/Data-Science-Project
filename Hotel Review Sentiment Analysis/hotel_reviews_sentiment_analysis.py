# -*- coding: utf-8 -*-

import pandas as pd
# Local directory
Reviewdata = pd.read_csv('/content/train.csv')
#Data Credit - https://www.kaggle.com/anu0012/hotel-review/data

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer

!pip install wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

Reviewdata.shape

Reviewdata.head()

Reviewdata.info()

Reviewdata.describe().transpose()

"""**DATA CLEANING / EDA**"""

### Checking Missing values in the Data Set and printing the Percentage for Missing Values for Each Columns ###

count = Reviewdata.isnull().sum().sort_values(ascending=False)
percentage = ((Reviewdata.isnull().sum()/len(Reviewdata)*100)).sort_values(ascending=False)
missing_data = pd.concat([count, percentage], axis=1,
keys=['Count','Percentage'])

print('Count and percentage of missing values for the columns:')

missing_data

Reviewdata['Is_Response'].unique()

#Removing columns
Reviewdata.drop(columns = ['User_ID', 'Browser_Used', 'Device_Used'], inplace = True)

# Apply first level cleaning
import re
import string

#This function converts to lower-case, removes square bracket, removes numbers and punctuation
def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)

# Let's take a look at the updated text
Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.Description.apply(cleaned1))
Reviewdata.head()

stp_words = stopwords.words('english')
def clean_review(review):
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview

Reviewdata['cleaned_description_new2'] = pd.DataFrame(Reviewdata['cleaned_description'].apply(clean_review))
Reviewdata.head()

# Apply a third round of cleaning
def text_clean_3(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned3 = lambda x: text_clean_3(x)

# Let's take a look at the updated text
Reviewdata['cleaned_description_new3'] = pd.DataFrame(Reviewdata['cleaned_description_new2'].apply(cleaned3))
Reviewdata.head()

new_df = Reviewdata[['Is_Response','cleaned_description_new3']].copy()
new_df

"""**Data Sampling**"""

new_df.Is_Response.value_counts()

import seaborn as sns

sns.countplot(new_df["Is_Response"])
plt.title("Count Plot of Response")
plt.show()

from sklearn.utils import resample

df_majority = new_df[new_df["Is_Response"]=='happy'] #majority class
df_minority = new_df[new_df["Is_Response"]=='not happy'] #minority class

# downsample the majority class
df_majority_downsampled = resample(df_majority,
                                 replace=False,
                                 n_samples=len(df_minority),
                                 random_state=1234)

new_df = df_majority_downsampled.append(df_minority)
new_df.head(2)

#Visualisation after downsampling
sns.countplot(new_df["Is_Response"])
plt.title("Count Plot of Response")
plt.show()

new_df.Is_Response.value_counts()

pos_reviews = new_df.loc[new_df['Is_Response'] == 'happy']
pos_reviews

neg_reviews = new_df.loc[new_df['Is_Response'] == 'not happy']
neg_reviews

pos_df = new_df.loc[new_df['Is_Response'] == 'happy']
pos_list = pos_df['cleaned_description_new3'].tolist()

neg_df = new_df.loc[new_df['Is_Response'] == 'not happy']
neg_list = neg_df['cleaned_description_new3'].tolist()

# using list comprehension
pos_list_to_string = ' '.join([str(elem) for elem in pos_list])
neg_list_to_string = ' '.join([str(elem) for elem in neg_list])

tokenizer = WhitespaceTokenizer()

stop = set(stopwords.words('english') + list(string.punctuation))
filtered_pos_list = [w for w in tokenizer.tokenize(pos_list_to_string) if w not in stop]
filtered_neg_list = [w for w in tokenizer.tokenize(neg_list_to_string) if w not in stop]
fd_pos = nltk.FreqDist(filtered_pos_list)
fd_neg = nltk.FreqDist(filtered_neg_list)

fd_pos.most_common(15)

fd_neg.most_common(15)

"""**DATA VISUALIZATION**"""

text = ' '.join([word for word in pos_reviews['cleaned_description_new3']])
plt.figure(figsize=(15,10), facecolor='None')
wordcloud = WordCloud(max_words=500, width = 1600, height = 800).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews\n', fontsize = 19)
plt.show()

text = ' '.join([word for word in neg_reviews['cleaned_description_new3']])
plt.figure(figsize=(15,10), facecolor='None')
wordcloud = WordCloud(max_words=500, width = 1600, height = 800).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews\n', fontsize = 19)
plt.show()

"""**MODEL TRAINING**"""

from sklearn.model_selection import train_test_split

Independent_var = new_df.cleaned_description_new3
Dependent_var = new_df.Is_Response

IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.2, random_state = 225)

print('IV_train :', len(IV_train))
print('IV_test  :', len(IV_test))
print('DV_train :', len(DV_train))
print('DV_test  :', len(DV_test))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")


from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

model.fit(IV_train, DV_train)


from sklearn.metrics import confusion_matrix

predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)

from sklearn import metrics
cm = confusion_matrix(predictions, DV_test)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = ['Positive', 'Negative'])

cm_display.plot()
plt.show()

"""**MODEL PREDICTION**"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy : ", accuracy_score(predictions, DV_test))
print("Precision : ", precision_score(predictions, DV_test, average = 'weighted'))
print("Recall : ", recall_score(predictions, DV_test, average = 'weighted'))

"""**TRYING ON NEW REVIEWS**"""

example = ["The rooms stink. They are not clean. I requested a non smoking room and both rooms smelled like smoke. Bathrooms were gross and bugs were everywhere! ! The door did not seem secure and was not evened out so bugs got in easily. The second room was full of gnats."]
result = model.predict(example)

if(result=='not happy'):
    print("NEGATIVE")
else:
    print('POSITIVE')

example = ["Rooms, concierge services and party scene were great.. right in middle of downtown San Diego .. the restaurants and clubs were very good.. will definitely come back"]
result = model.predict(example)

if(result=='not happy'):
    print("NEGATIVE")
else:
    print('POSITIVE')

example = ["I am not happy with the service given by the staff. They are unfriendly!"]
result = model.predict(example)

if(result=="not happy"):
    print("NEGATIVE")
else:
    print('POSITIVE')

example = ["Comfortable room… location are very convenient for eating, sightseeing & shopping. We arrived early, the room was available"]
result = model.predict(example)

if(result=="not happy"):
    print("NEGATIVE")
else:
    print('POSITIVE')

example = ["The room only got shower gel, no other toiletries, not even slipper, which to me such a basic thing to have in a hotel. The cup smells bad especially when you put hot water in it."]
result = model.predict(example)

if(result=='not happy'):
    print("NEGATIVE")
else:
    print('POSITIVE')