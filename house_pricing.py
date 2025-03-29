import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = fetch_california_housing()
dataset = pd.DataFrame(df.data, columns=df.feature_names)
dataset['Outcome'] = df.target
# print(dataset.head())
# sns.pairplot(dataset, hue='Outcome', diag_kind="kde")
# plt.savefig("nigga")

# checking for mssing values

# print(dataset.isnull().sum())

# correlation

correlation = dataset.corr()

# creating a heatmap

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size' : 8}, cmap='Blues')
plt.savefig('heatmap')



X = dataset.drop(columns=['Outcome'], axis=1)
Y = dataset['Outcome']


# splitting first
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# standardize the data

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# making the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# testing the model 
X_test_predictions = model.predict(X_test)
# score = accuracy_score(X_test_predictions, Y_test)
# print(score)
# this shit is wrong because the accuracy_score is meant for classification model and not reggression models so in this case we use
# old ass mean squared error
score = r2_score(Y_test, X_test_predictions)
score2 = mean_absolute_error(Y_test, X_test_predictions)
print(f"R2 score = {score} and mean absolute error = {score2}")
