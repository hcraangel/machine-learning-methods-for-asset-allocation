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
Fea_ext = DataExtend.Data_extend(Fea, shift_lag = 1 + 1)

################################# SVR #################################
data_set = Fea_ext.join(Ret['GLD']).dropna()
train_set = data_set.loc[:'2010-12-31',:]
test_set = data_set.loc['2011-01-31':,:]

X_train = train_set.iloc[:,:-1].values
y_train = train_set.iloc[:,-1].values
y_train = y_train.reshape(len(y_train),1)

X_test = test_set.iloc[:,:-1].values
y_test = test_set.iloc[:,-1].values
y_test = y_test.reshape(len(y_test),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C = 500, epsilon = 0.01  )
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_train = regressor.predict(X_train)
y_pred_train = sc_y.inverse_transform(y_pred_train)
y_pred_train = y_pred_train.reshape(len(y_pred_train),1)

# 
y_pred_test = regressor.predict(X_test)
y_pred_test = sc_y.inverse_transform(y_pred_test)
y_pred_test = y_pred_test.reshape(len(y_pred_test),1)

y_train = sc_y.inverse_transform(y_train)

plt.plot(y_pred_train,y_train,'.')
plt.plot(y_pred_test,y_test,'.')


import statsmodels.api as sm
result1 = sm.OLS(y_pred_train,y_train).fit()
result2 = sm.OLS(y_pred_test,y_test).fit()
result1.summary()
result2.summary()


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(y_pred_train, y_train)

# Predicting the Test set results
y_ols_pred = regressor2.predict(y_pred_test)

plt.plot(y_ols_pred,y_test,'.')




plt.plot(y_train)
