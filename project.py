import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

#from gensim.models import Word2Vec, KeyedVectors

st.title("Emotion Classification")


df = pd.read_csv('Twitter_Emotion_Datasetab.csv')

text = df['tweet']
y = df['label'].values

count_vect = CountVectorizer()
X_count = count_vect.fit_transform(text)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)

classifier = LogisticRegression(multi_class='multinomial', penalty='l2', solver='newton-cg', C= 1.623776739188721)
classifier.fit(X_tfidf, y)

sentence = st.text_input('Input your sentence here:') 

text_new =[sentence]
X_new_counts = count_vect.transform(text_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = classifier.predict(X_new_tfidf)
prediction_proba = classifier.predict_proba(X_new_tfidf)

if sentence:
    st.write(prediction[0])

    st.subheader('Class labels and index number')
    st.write(classifier.classes_)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

