import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
dataset = pd.read_csv('loans.csv')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



# Handling Missing values

# sns.displot(dataset['LoanAmount'])
# plt.savefig('Loan')
dataset.fillna({'LoanAmount' : dataset['LoanAmount'].median()}, inplace=True)
dataset.fillna({'Loan_Amount_Term' : dataset['Loan_Amount_Term'].mode()[0]}, inplace=True)
dataset.dropna(subset=['Married'], inplace=True)
dataset.fillna({'Self_Employed' : dataset['Self_Employed'].mode()[0]}, inplace=True)
dataset.fillna({'Gender' : dataset['Gender'].mode()[0]}, inplace=True)
dataset.fillna({'Dependents' : dataset['Dependents'].mode()[0]}, inplace=True)
dataset.fillna({'Credit_History' : dataset['Credit_History'].mode()[0]}, inplace=True)
# print(dataset.isnull().sum())

# Handling the imbalance in the data
# yes_loan = dataset[dataset['Loan_Status'] == 'Y']
# no_loan = dataset[dataset['Loan_Status'] == 'N']
# yes_loan_sample = yes_loan.sample(n = 192)
# df = pd.concat([yes_loan_sample, no_loan])
df = dataset

# Label Encoding


categorical_var = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_var:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])


# taking care of the propblem of 3+ in the dependents columns
df['Dependents'] = df['Dependents'].replace("3+", "3").astype(int)
df = df.drop(columns=['Loan_ID'])

# making a pairplot
# sns.pairplot(df, hue="Loan_Status", diag_kind="kde")
# plt.savefig("nigga")

# standardizing


# splitting time
X = df.drop(columns=['Loan_Status'])
Y = df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=8)
smote = SMOTE(random_state=8)
X_train, Y_train = smote.fit_resample(X_train, Y_train)



# training the model

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# cheking the accuracy of the model

X_test_predicted = model.predict(X_test) 
score = accuracy_score(X_test_predicted, Y_test)
print(score)

# first the score was getting too bad then found out that original method of handling the imbalance was bad as it 
# removed the data and it hurt the performance here so i used SMOTE to oversample the original short data