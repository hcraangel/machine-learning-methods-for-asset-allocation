import numpy as np
import pandas as pd
import datetime as dt
import os
from matplotlib import pyplot as plt

################################# Import data #################################

os.chdir('F:\Capstone\Github\machine-learning-methods-for-asset-allocation')

## Dataframe processe
Ret = pd.read_csv('returns.csv')
Fea = pd.read_csv('features.csv')

os.chdir('F:\Capstone\Capstone\Codes\main')


Fea.index = pd.to_datetime(Fea['date'])
Ret.index = pd.to_datetime(Ret['date'])
del Fea['date'], Ret['date']

# strip the spaces
Ret.columns = [x.strip() for x in Ret.columns]
Fea.columns = [x.strip() for x in Fea.columns]

################################# Extend data #################################
import DataExtend
Fea_ext1 = DataExtend.Data_extend(Fea, shift_lag = 1 + 1)
Fea_ext2 =  Fea_ext1.drop(columns = list(Fea.columns))
Fea_ext2[['GDP','LIBOR12M','CFNAI','CU']] = Fea_ext1[['GDP','LIBOR12M','CFNAI','CU']]
Fea_ext = Fea_ext2['1950-01-31':].dropna(axis = 1)

## Build dependent and independent
dataset = Fea_ext.join(Ret['SPY']).dropna()
X = dataset.iloc[:,:-1]
y_org = dataset.iloc[:,-1]
# categorize
sym_para_L = 1
sym_para_r = 0
bin = [y_org.min() - 0.001, 0,
       y_org.mean() + y_org.std(),y_org.max()]
y = pd.cut(y_org, bin, labels = ['short','hold','long'])


y_org.hist(bins = 50)
sum(y_org >0)/len(y_org)

bin = [y_org.min() - 0.001, 0.00 ,y_org.max()]
y = pd.cut(y_org, bin, labels = ['short','long'])
(y == 'long').sum()

################################# XG boost #################################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

train_start = 360
test_range = 36
X_train = X[train_start:-test_range]
y_train = y[train_start:-test_range]
X_test = X[-test_range:]
y_test = y[-test_range:]

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

yy = pd.DataFrame(y_test)
yy['pred'] = pd.Series(y_pred, index = yy.index)

[y_pred == y_test]


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()



sum(y_pred == 'short')
sum(y_pred == 'long')
