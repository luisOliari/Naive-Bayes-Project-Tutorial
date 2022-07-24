from utils import db_connect
engine = db_connect()

# your code here

# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import pickle
import nltk
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB

# Read csv

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

# mirando el dataset:
df_raw.head(10)

# viendo información del dataset
df_raw.info()

#Transformar el Dataset
df_transf = df_raw.copy()
df_transf = df_transf.drop('package_name', axis=1)

# trabajamos sobre la columna review
df_transf['review'] = df_transf['review'].str.strip()
# elimina espacio libre al principio y al final

# column review to lower case
df_transf['review'] = df_transf['review'].str.lower()

stop = stopwords.words('english')

def clean_words(review):
    if review is not None:
        words = review.strip().split()
        new_words = []
        for word in words:
            if word not in stop:new_words.append(word)
        Result = ' '.join(new_words)    
    else:
        Result = None
    return Result

df_transf['review'] = df_transf['review'].apply(clean_words)

# llamamos a la transf.copy = df
df = df_transf.copy()

# Split data frame
X = df['review']
y = df['polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)

# Vectorize text reviews to numbers: 
vect_tfidf = TfidfVectorizer()
text_vec_tfidf = vect_tfidf.fit_transform(X_train)

vect.get_feature_names_out() #da nombres de las columnas


clf_2 = MultinomialNB()

clf_2.fit(text_vec_tfidf, y_train)

pred_2 = clf_2.predict(vect_tfidf.transform(X_test))
print(classification_report(y_test, pred_2))

# Acá todo junto en un pipeline
text_clf_2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf_2.fit(X_train, y_train)

# Check Result
y_pred = text_clf_2.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='weighted')

print('Naive Bayes Train Accuracy = ',metrics.accuracy_score(y_train,text_clf_2.predict(X_train)))
print('Naive Bayes Test Accuracy = ',metrics.accuracy_score(y_test,text_clf_2.predict(X_test)))

# Randomized search to select hyperparameters

n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = RandomizedSearchCV(text_clf_2, parameters, n_iter = n_iter_search)
gs_clf.fit(X_train, y_train)

gs_clf.best_params_

print('Naive Bayes Train Accuracy (grid random search) = ',metrics.accuracy_score(y_train,gs_clf.predict(X_train)))
print('Naive Bayes Test Accuracy (grid random search) = ',metrics.accuracy_score(y_test,gs_clf.predict(X_test)))

text_clf_count_vect = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
text_clf_count_vect.fit(X_train, y_train)


n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': (1e-2, 1e-3)}
gs_count_vect = RandomizedSearchCV(text_clf_count_vect, parameters, n_iter = n_iter_search)
gs_count_vect.fit(X_train, y_train)

gs_count_vect.best_params_

print('Naive Bayes Train Accuracy (grid random search) = ',metrics.accuracy_score(y_train,gs_count_vect.predict(X_train)))
print('Naive Bayes Test Accuracy (grid random search) = ',metrics.accuracy_score(y_test,gs_count_vect.predict(X_test)))

y_pred_mejor = gs_clf.predict(X_test)

print(classification_report(y_test, y_pred_mejor))



