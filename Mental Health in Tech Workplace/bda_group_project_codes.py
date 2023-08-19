# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('cool')
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('survey.csv')
df.head(3)

"""# EDA"""

df.shape

df.describe()

df.info()

"""We can see that *comments* have many null values and *Timestamp* column is not useful for our analysis thus we will drop these two column."""

#drop column Timestamp and comments
df.drop(['Timestamp','comments'], axis=1, inplace=True)

#analyse values in column Age
df.Age.unique()

"""It is impossible to have age in negative values, in hundreds and below 15 years-old at workplace thus we'll be removing these outliers by setting the min age to be 15 because 15 is the legal age to work and as for max age we'll decide based on boxplot below."""

index = df[(df['Age'] > 70)|(df['Age'] < 15)].index
df.drop(index, inplace=True)

#Boxplot and histogram to identify outliers
import plotly.express as px
px.histogram(df, x="Age", marginal="box", title='Respondents Age With Outliers')

#set the max age to be 47 years-old
index = df[(df['Age'] > 47)|(df['Age'] < 15)].index
df.drop(index, inplace=True)

import plotly.express as px
px.histogram(df, x="Age", marginal="box", title='Respondents Age Without Outliers')

#analyse column country
df.Country.value_counts()

"""Most of the respondents are Americans and there are many countries with few number of respondents like 1,2 and 3. It is unfair to conclude that majority workers in USA are having mental health issue in workplace thus we will drop column *Country* as well as column *state*"""

#drop column Country and state
df.drop(['Country','state'], axis=1, inplace=True)

df.info()

"""Again we'll have a look at our dataframe info. All columns have non-null values except for *self_employe*d and *work_interfere* so next we'll drop rows with null values."""

#drop null values
df.dropna(inplace=True)

#again, we'll confirm all null values have been removed
df.isnull().sum()

#check for duplicate rows
duplicate = df[df.duplicated()]
print('There are {} duplicate rows'.format(len(duplicate)))

#remove duplicates
df.drop_duplicates(inplace=True)

#analyse column Gender
df.Gender.value_counts()

"""There are so many answers for the same categories hence we'll clean *Gender* column to only have three categories which are *female, male* and *others*"""

#lowercase the values in column Gender
df.Gender = df.Gender.str.lower()

#assign the answers into their respective categories
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]
female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
other = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid",
         "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
         "female (trans)", "queer", "ostensibly male, unsure what that really means", "p", "a little about you"]

df.Gender.loc[df.Gender.isin(male)] = 'male'
df.Gender.loc[df.Gender.isin(female)] = 'female'
df.Gender.loc[df.Gender.isin(other)] = 'others'

df.Gender.value_counts()

df.count()

"""# DATA SCIENCE QUESTIONS

**Descriptive Analysis**

1.   What is the average age of employees based on gender?
"""

series = df.groupby('Gender')['Age'].mean()
to_df = pd.DataFrame(series)
table_gender = to_df.style.background_gradient(cmap=cmap)
table_gender

"""2. How many employers are primarily tech companies?"""

count_tech=df.tech_company.value_counts()

plt.figure(figsize=(7,6))

explode = [0.1, 0]
colors=['#ADC8FA','#FFB799']
label=['Yes', 'No']

plt.pie(count_tech, labels=label, explode=explode, shadow=True, colors=colors, autopct='%1.1f%%',)
plt.legend(title = "Tech Company")
plt.title('Employers that Primarily Technology Companies', y=1.12, fontsize=12, fontweight='bold')
plt.show()

"""**Exploratory Analysis**

1.   Is an employee willing to discuss a mental health issue in an interview for those who think that discussing it with the employer would not have negative consequences and are eager to discuss it with the supervisor?
"""

df_emp = df[df['mental_health_consequence']=='No']
df_sup = df_emp[df_emp['supervisor']=='Yes']

sns.countplot(y=df_sup.mental_health_interview, color='#ADC8FA')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('\nNumber of Employees');
plt.title('Mental Health Issue in Interview\n', fontsize=15, horizontalalignment='center')
plt.show()

"""2. Are the employees easy to take a medical leave for a mental health condition?"""

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
eda_percentage = df['leave'].value_counts(normalize = True).rename_axis('leave').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'leave', y = 'Percentage', data = eda_percentage, palette='Blues')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Leave for Mental Health provided to the Employees\n', fontsize=18, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(df['leave'], hue = df['treatment'], palette='Blues')
plt.title('Leave for Mental Health provided to the Employees\n', fontsize=18, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

"""3. Do many employers provide mental health benefits?"""

df.benefits.value_counts()

fig = plt.figure(figsize=(15,6))

ax3 = fig.add_subplot(1,2,1)
sns.countplot(x ='benefits', data = df, palette='pastel').set_title("Mental Health Benefits for Employees", y=1.12, fontsize=12, fontweight='bold')

ax4 = fig.add_subplot(1,2,2)
sns.countplot(x ="benefits", hue = "treatment", data = df, palette = 'pastel', ax=ax4).set_title("Correlation of Mental Health Benefits and Seeking Treatment", y=1.12, fontsize=12, fontweight='bold')

"""**Inferential Analysis**

1.   Do juniors and middles typically need mental health treatment more than seniors?
"""

df_temp = df.copy()

df_temp.loc[df['Age'].between(19,30), 'age_group'] = 'Juniors'
df_temp.loc[df['Age'].between(31,40), 'age_group'] = 'Middles'
df_temp.loc[df['Age']>40, 'age_group'] = 'Seniors'

df_need = df_temp[df_temp['treatment']=='Yes']

sns.countplot(y=df_need.age_group, color='#ADC8FA')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('\nNumber of Employees');
plt.title('Age Group who Seeks Mental Health Treatment\n', fontsize=15, horizontalalignment='center')
plt.show()

"""2. Do employees from tech companies tend to seek mental health treatment?"""

z = df[(df["tech_company"] == 'Yes')].groupby(['treatment'])['treatment'].count()
z.plot(kind='bar', figsize=(8,5), color=['#ADC8FA','#FFB799','#ACF7B8'])
plt.title("Employees from Tech Companies that Seek Mental Health Treatment")
plt.ylabel('Number Of Employees ')
plt.xlabel('Mental Health Consequence')

"""3.   Do many employees feel their employer takes mental health as seriously as physical health if it is somewhat easy and very easy for them to take medical leave for a mental health condition?


"""

df_leave = df[(df['leave']=='Somewhat easy') | (df['leave']=='Very easy')]

sns.countplot(y=df_leave.mental_vs_physical, color='#ADC8FA')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('\nNumber of Employees');
plt.title('Employer Takes Mental Health as Seriously as Physical Health\n', fontsize=13, horizontalalignment='center')
plt.show()

"""4. Are the employees more likely to develop mental illness if there is a family history of it?"""

df[(df['family_history'] == 'Yes') & (df['treatment'] == 'Yes')].shape

df[(df['family_history'] == 'Yes') & (df['treatment'] == 'No')].shape

#show in Pie chart
plt.figure(figsize=(7,9))
treatment = ['Yes','No']
value_count = [344,81]

plt.pie(value_count,labels=treatment,colors=colors,autopct="%2.2f%%", explode=explode, shadow=True)
plt.legend(title='Seeking Treatment')
plt.title('Employees Seeking Treatment Based on Family History', y=1.12, fontsize=12, fontweight='bold')
plt.show()

"""# Data Preparation"""

#check values for each column
list_col=['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']

for col in list_col:
    print('{} :{} \n' . format(col, df[col].unique()))

#label encoding all columns except for column Age cause Age already in int data type
from sklearn.preprocessing import LabelEncoder

object_cols = ['Gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work', 'tech_company',
               'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
               'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
               'mental_vs_physical', 'obs_consequence']

label_encoder = LabelEncoder()
for col in object_cols:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])

df.benefits.value_counts()

#data splitting
from sklearn.model_selection import train_test_split

X = df.drop(['treatment'], axis = 1)
y = df.treatment

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size = 0.2, random_state = 42)

#check for training and testing shape
X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""**PREDICTIVE ANALYSIS**

# Naiive Bayes (NB)
"""

#model training
#from sklearn.naive_bayes import GaussianNB     #-> features are continous (normal distribution)
from sklearn.naive_bayes import BernoulliNB     #-> assumes all features are binary, suitable for discrete data, just like multinmialNB but the labels are boolean/binary
#from sklearn.naive_bayes import ComplementNB   #-> suited for imbalanced data sets
#from sklearn.naive_bayes import MultinomialNB

import random
np.random.RandomState(5)

#model_nb = GaussianNB()
model_nb = BernoulliNB(fit_prior=True)
#model_nb = ComplementNB()
#model_nb = MultinomialNB()

model_nb.fit(X_train, y_train)
nb_pred = model_nb.predict(X_test)

# Predict Test Set (Naiive Bayes)
nb_pred

#accuracy score for Naiive Bayes
from sklearn.metrics import accuracy_score

score=accuracy_score(y_test,nb_pred)
print('Naiive Bayes accuracy is {}%' .format(round(score*100,2)))

#classification report of NB
from sklearn.metrics import classification_report

print(classification_report(y_test,nb_pred))

from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: {}%'.format(round(precision_score(y_test, nb_pred)*100,2)))
print('Recall: {}%'.format(round(recall_score(y_test, nb_pred)*100,2)))
print('f1: {}%'.format(round(f1_score(y_test, nb_pred)*100,2)))

pip install shap

import shap

# train nb model
model_nb_shap = model_nb.fit(X_train, y_train)

def f(x):
    return model_nb_shap.predict(x)

# compute SHAP values
explainer = shap.Explainer(f, X_train)
shap_values = explainer(X_train)

shap.plots.bar(shap_values, max_display=10)

shap.plots.beeswarm(shap_values, max_display=10)

"""# Deep Convolutional Neural Network (CNN)"""

#from sklearn.linear_model import Perceptron#

#perceptron = Perceptron()
#perceptron.fit(X_train, y_train)
#Y_pred = perceptron.predict(X_test)
#acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
#acc_perceptron

from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense

try:
    import scikeras
except ImportError:
    !python -m pip install scikeras

from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras

#from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim = 22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(learning_rate = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 15, 20]

# make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_standardized = scaler.transform(X)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=None), verbose = 10)
grid_results = grid.fit(X_standardized, y)

#get the best parameters for GridSearchCV
print("\n Best parameters:", grid.best_params_)

cnn_pred = grid.predict(X_standardized)

result=accuracy_score(y, cnn_pred)
print('CNN accuracy is {}%\n' .format(round(result*100,2)))
#print(accuracy_score(y, cnn_pred))
print(classification_report(y, cnn_pred))

y_test.shape, cnn_pred.shape,y.shape

print('Precision: {}%'.format(round(precision_score(y, cnn_pred)*100,2)))
print('Recall: {}%'.format(round(recall_score(y, cnn_pred)*100,2)))
print('f1: {}%'.format(round(f1_score(y, cnn_pred)*100,2)))

#batch_size2 = [40]
#epochs2 = [10]

#param_grid2 = dict(batch_size=batch_size2, epochs=epochs2)

#grid2 = GridSearchCV(estimator = model, param_grid = param_grid2, cv = KFold(random_state=None), verbose = 10)
#grid_results2 = grid2.fit(X_standardized, y)       # feature_names = X_standardized.columns.tolist()

def g(x):
    return grid_results.predict(x)

# compute SHAP values
explainer2 = shap.Explainer(g, X_train)
shap_values2 = explainer(X_train)

#shap.plots.bar(shap_values, max_display=10)
shap.plots.beeswarm(shap_values2, max_display=10)

"""# NB + CNN + MLP"""

#Define estimators
from sklearn.ensemble import StackingClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

best_batch_size = [40]
best_epochs = [10]

# make a dictionary of the grid search parameters
new_param_grid = dict(batch_size=best_batch_size, epochs=best_epochs)
new_grid = GridSearchCV(estimator = model, param_grid = new_param_grid, cv = KFold(random_state=None), verbose = 10)


estimator_list = [
    ('nb' , model_nb),      #NB
    ('cnn', new_grid)]      #CNN

# Build stack model
stack_model = StackingClassifier( estimators=estimator_list, final_estimator=MLPClassifier(solver="adam", shuffle=True, activation="logistic")
                                 #final_estimator=LogisticRegression(solver='lbfgs', random_state=6, verbose=10, l1_ratio=1)
                                 #final_estimator=SVC(gamma='scale', C=1, random_state=62, probability=True, kernel='rbf')
)

# Train stacked model
stack_model.fit(X_train, y_train)

# Make predictions
y_train_pred = stack_model.predict(X_train)
y_test_pred = stack_model.predict(X_test)

# Training set model performance
stack_model_train_accuracy = accuracy_score(y_train, y_train_pred)

print("\n\nTraining accuracy for stacked model: {}". format(stack_model_train_accuracy))

# Predict Test Set (Stacked Model)
y_test_pred

print("Accuracy of multiple models is {}%\n".format(round(accuracy_score(y_test, y_test_pred)*100,2)))
print(classification_report(y_test,y_test_pred))

print('Precision: {}%'.format(round(precision_score(y_test, y_test_pred)*100,2)))
print('Recall: {}%'.format(round(recall_score(y_test, y_test_pred)*100,2)))
print('f1: {}%'.format(round(f1_score(y_test, y_test_pred)*100,2)))

model_stack_shap=stack_model.fit(X_train, y_train)

def h(x):
    return model_stack_shap.predict(x.astype(int))

# compute SHAP values
explainer3 = shap.Explainer(h, X_train)
shap_values3 = explainer3(X_train)

shap.plots.beeswarm(shap_values3, max_display=10)