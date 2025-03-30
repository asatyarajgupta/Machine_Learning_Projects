import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('winequality-red.csv')
# print(df['quality'].value_counts())

# creating a heatmap

# corr = df.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corr, cbar=True, annot_kws={'size' : 8}, fmt=".1f", annot=True, cmap='Blues', square=True)
# plt.savefig('heatmap2')

# train test split
X = df.drop(columns=['quality'])
Y = df['quality'].apply(lambda y_value: 1 if y_value >= 6 else 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# traing the model

model = RandomForestClassifier()
model.fit(X_train, Y_train)

# checking the accuracy of the model

predicted_X_test = model.predict(X_test)
score = accuracy_score(predicted_X_test, Y_test)

print(score*100, "%")

# predicting something

A = [[10.8,0.32,0.44,1.6,0.063,16.0,37.0,0.9985,3.22,0.78,10.0]]
B = model.predict(A)
if (B == 1):
    print('The wine is pretty good')
else:
    print("This wine is not good")
    print(B)
