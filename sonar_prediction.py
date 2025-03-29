# we are gonna use logistic regression for this model because it really works well for a binary classification problem


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv("~/code/machine_learning/sonar.csv")


# label encoding
encoder = LabelEncoder()
labels = encoder.fit_transform(df.iloc[:,60])
df[61] = labels
# print(df.iloc[:, 60 : 62])
X = df.drop(df.columns[60 : 62], axis=1)
Y = df.iloc[:, 61]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


# Standaradizing the data (gave lesser score with than wihtout)
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)

# training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# scoring the model
X_test_prediction = model.predict(X_test)
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print(test_data_accuracy, " ", train_data_accuracy)

# making it an actual application
input_thing =[[0.0412,0.1135,0.0518,0.0232,0.0646,0.1124,0.1787,0.2407,0.2682,0.2058,0.1546,0.2671,0.3141,0.2904,0.3531,0.5079,0.4639,0.1859,0.4474,0.4079,0.5400,0.4786,0.4332,0.6113,0.5091,0.4606,0.7243,0.8987,0.8826,0.9201,0.8005,0.6033,0.2120,0.2866,0.4033,0.2803,0.3087,0.3550,0.2545,0.1432,0.5869,0.6431,0.5826,0.4286,0.4894,0.5777,0.4315,0.2640,0.1794,0.0772,0.0798,0.0376,0.0143,0.0272,0.0127,0.0166,0.0095,0.0225,0.0098,0.0085]]
# input_array = np.asarray(input_thing)
# print(input_array)
input_train_prediction = model.predict(input_thing)
if (input_train_prediction[0] == 0) :
    print("its a mine nigga")
else:
    print("dw its rock lil nigga")