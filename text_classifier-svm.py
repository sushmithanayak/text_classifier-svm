#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("bbc-text.csv")


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


df.isnull().sum()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X = df['text']


# In[8]:


y = df['category']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[10]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# In[11]:


# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# In[12]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[35]:


import nltk
nltk.download()


# In[13]:


# Removing stop words
# Stemming Code
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)


# In[14]:


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# In[15]:


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')


# In[16]:


# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary
text_clf = Pipeline([('vect', stemmed_count_vect),('tfidf',TfidfTransformer()), ('clf',LinearSVC())])


# In[17]:


text_clf.fit(X_train, y_train)


# In[18]:


predictions = text_clf.predict(X_test)


# In[19]:


#checking the performance
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score


# In[20]:


print(confusion_matrix(y_test,predictions))


# In[21]:


print(classification_report(y_test,predictions))


# In[22]:


print(accuracy_score(y_test,predictions))
