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
fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10))
snp.countplot(x='Survived', data=train_data, hue='Sex', ax=axes[0])
snp.countplot(x='Survived', data=train_data, hue='Pclass', ax=axes[1])
snp.distplot(train_data['Age'], kde=False, bins=30, ax=axes[2])

# since we have semi-acceptance data as we have null values present.
# Here we will clear and refine data
plt.show()