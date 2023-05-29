#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[4]:


tweet_df = pd.read_csv('/content/train.csv')


# In[5]:


tweet_df.head()


# In[6]:


tweet_df.info()


# In[7]:


# printing random tweets 
print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[8]:


#creating a function to process the data
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)


# In[10]:


import nltk
nltk.download('punkt')


# In[11]:


tweet_df.tweet = tweet_df['tweet'].apply(data_processing)


# In[12]:


tweet_df = tweet_df.drop_duplicates('tweet')


# In[13]:


lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmarizer.lemmatize(word) for word in data]
    return data


# In[16]:


import nltk
nltk.download('wordnet')


# In[18]:


import nltk
nltk.download('omw-1.4')


# In[19]:


tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizer.lemmatize(x))


# In[20]:


print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[21]:


tweet_df.info()


# In[22]:


tweet_df['label'].value_counts()


# In[23]:


#DATA-VISUALISATION
fig = plt.figure(figsize=(5.5,5))
sns.countplot(x='label', data = tweet_df)


# In[27]:


fig = plt.figure(figsize=(8,8))
colors = ("pink", "yellow")
wp = {'linewidth':2, 'edgecolor':"white"}
tags = tweet_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie',autopct = '%1.1f%%', shadow=True, colors = colors, startangle =90, 
         wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of BullyCases')


# In[28]:


non_hate_tweets = tweet_df[tweet_df.label == 0]
non_hate_tweets.head()


# In[29]:


text = ' '.join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1200, height=600).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 19)
plt.show()


# In[30]:


neg_tweets = tweet_df[tweet_df.label == 1]
neg_tweets.head()


# In[31]:


text = ' '.join([word for word in neg_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1200, height=600).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in hate tweets', fontsize = 19)
plt.show()


# In[32]:


vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet'])


# In[33]:


feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 30 features: \n{}".format(feature_names[:30]))


# In[34]:


vect = TfidfVectorizer(ngram_range=(1,3)).fit(tweet_df['tweet'])


# In[37]:


feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 25 features: \n{}".format(feature_names[:25]))


# In[38]:


#MODEL BUILDING


# In[39]:


X = tweet_df['tweet']
Y = tweet_df['label']
X = vect.transform(X)


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)


# In[44]:


print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# In[45]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuarcy: {:.2f}%".format(logreg_acc*100))


# In[46]:


print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))


# In[49]:


style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()


# In[50]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[51]:


param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver' :['newton-cg', 'lbfgs','liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
grid.fit(x_train, y_train)
print("Best Cross validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)


# In[52]:


y_pred = grid.predict(x_test)


# In[53]:


logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[54]:


print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


# In[ ]:




