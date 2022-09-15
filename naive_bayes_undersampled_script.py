from random import Random
import re
import string
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

df = pd.read_json('training_json_file.json', orient='records')
print(df.head())

dictionary = {1700: 0, 1800: 1, 1900: 2, 2000: 3}

df['label']= df['label'].map(dictionary)

resampled_df = df.groupby('label').apply(lambda x: x.sample(135)).reset_index(drop=True)
print(resampled_df.info)
print(resampled_df['label'].value_counts())

tf = TfidfVectorizer()
X_tf = tf.fit_transform(resampled_df['cleaned_html'])
X_tf = X_tf.toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tf, resampled_df['label'].values, test_size=0.25)

'''
naiveBayes = GaussianNB()
naiveBayes.fit(X_train, y_train)
y_pred = naiveBayes.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=naiveBayes.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=naiveBayes.classes_)
disp.plot()
plt.show()
'''

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred_rfc))
print(accuracy_score(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))

