# we use SVM for thsi model
# Checking Feature Correlations (Scatter Plots)
# Linear separability: If points of different classes are clearly separated (e.g., along a straight line), Logistic Regression might work well.
# Overlapping points: If the two classes are highly mixed, a more complex model (like SVM with an RBF kernel or Neural Networks) may be better.
# Strong correlations: If two features form a distinct pattern (e.g., a straight or curved trend), they may be good predictors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("~/code/machine_learning/diabetes.csv")
# chekcing waht kind it is since it is overlapping and no linearity it is done by svm as it be easily sperated by hyperline if this binary classification was a bit linear then this shit would be done by lgoistic reggression
# sns.pairplot(df, hue="Outcome", diag_kind="kde")
# plt.savefig("nigga")


# chekcing for missing values

# print(df.isnull().sum())

# checking for imbalance in the data

# print(df['Outcome'].value_counts())
diabetic = df[df['Outcome'] == 1]
non = df[df['Outcome'] == 0]
balanced_non = non.sample(n=268)
dataset = pd.concat([balanced_non, diabetic])
# print(dataset['Outcome'].value_counts())

# Cheking for standardizing

# print(dataset.describe())
scale = StandardScaler()
X = dataset.drop(columns=['Outcome'], axis=1)
Y = dataset['Outcome']
X = scale.fit_transform(X)

# training the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2) #startify to make sure equal distribution of Y
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# measuring the model
X_train_predicted = model.predict(X_train)
score = accuracy_score(X_train_predicted, Y_train)

# making an application
input = [[10,168,74,0,0,38,0.537,34]]
print(model.predict(input))