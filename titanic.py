# import essential libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

# fetch test and train data from csv files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# checking head of the test data
print(train_data.head())

# analysing some information from train data
cf.go_offline()
snp.set_style('whitegrid')
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
snp.countplot(x='Survived', data=train_data, hue='Sex', ax=axes[0][0])
snp.countplot(x='Survived', data=train_data, hue='Pclass', ax=axes[0][1])
snp.distplot(train_data['Age'], kde=False, bins=30, ax=axes[1][0])
snp.boxplot(x='Pclass', y='Age', data=train_data, ax=axes[1][1])

# assign dummies variable for male and Embarked as our model can read them as an input
sex = pd.get_dummies(data=train_data['Sex'], drop_first=True)
embarked = pd.get_dummies(data=train_data['Embarked'], drop_first=True)
train_data = pd.concat([train_data, sex, embarked], axis=1)

# since we have semi-acceptance data as we have null values present.
# Here we will clear and refine data
# will drop cabin feature from train and test data as it contains many null values.
train_data.drop(['Cabin','Name', 'PassengerId', 'Sex', 'Embarked'], axis=1, inplace=True)

# will fill null values of feature by calculating average age of people resides in a Passenger class.
def mean_age(col):
	Age, Pclass= col[0], col[1]
	if pd.isnull(Age):
		if Pclass == 1:
			return cal_age(1)
		elif Pclass == 2:
			return cal_age(2)			
		else :
			return cal_age(3)
	else:
		return Age

def cal_age(num):
	return train_data[train_data['Pclass'] == num]['Age'].mean()

train_data['Age'] = train_data[['Age', 'Pclass']].apply(mean_age, axis=1)
plt.show()