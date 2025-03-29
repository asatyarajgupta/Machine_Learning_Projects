import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('train.csv')
# alot od the values in the dataset is empty so we are gonna use fillna to place spaces in place of them
# print(dataset.isnull().sum())

dataset.fillna('', inplace=True)
# print(dataset.isnull().sum())
dataset['content'] = dataset['title'] + " " + dataset['author']
X = dataset.drop(columns=['label'])
Y = dataset['label']
# print(Y)

nltk.download('stopwords')


# stemming proccess

stemmer = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content
X['content'] = X['content'].apply(stemming)


# featuring extracting
A = X['content'].values
B = Y.values

vectorizer = TfidfVectorizer()
A = vectorizer.fit_transform(A)


# splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(A, B, test_size=0.2, random_state=8, stratify=B)



# model making using logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# checking the accuracy of the model

predicted_A = model.predict(X_test)
score = accuracy_score(predicted_A, Y_test)
print(score)