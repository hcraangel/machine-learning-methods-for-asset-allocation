import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
import os

os.chdir('C:/2018 Fall Quarter/Codes and Data')

Ret = pd.read_csv('returns.csv')
Fea = pd.read_csv('features.csv')

# Set the index to date
Fea.index = Fea['date']
Ret.index = Ret['date']
del Fea['date'], Ret['date']

# delete all spaces in the column names
Ret.columns = [x.strip() for x in Ret.columns]
Fea.columns = [x.strip() for x in Fea.columns]
# merge features and returns
All = pd.merge(Ret, Fea, on = 'date')
# only select SPY
SPY_raw = All.drop(columns = ['DIA','IEF','GLD']).copy()

SPY1 = SPY_raw.dropna()

# Plot correlation matrix
import seaborn as sns
f, ax = plt.subplots(figsize = (15,7))
spy_corr = SPY1.corr().abs()

sns.heatmap(spy_corr,
            mask=np.zeros_like(spy_corr, dtype=np.bool), 
            square=True, ax=ax)
    
############################################################
############################################################
############################################################
############################################################
## Principal Component Analysis
# standardize the data
X = SPY1.drop('SPY', axis = 1)
y = SPY1['SPY']

# Do PCA
from sklearn import preprocessing
X_stand = preprocessing.scale(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 'mle')         # Set component number as auto
X_PCA = pca.fit_transform(X_stand)      # Get new features from PCA

## Do regression
import statsmodels.api as sm

# Regression on new features (principal components)
model = sm.OLS(y,X_PCA)
result = model.fit()
result.summary()

# Regression on original features
model2 = sm.OLS(y,X)
result2 = model2.fit()
result2.summary()


# individual X
# BDEBT LIBOR12M ICL


X_temp = X[['LIBOR12M','ICL']]
model3 = sm.OLS(y,X_temp)
result3 = model3.fit()

result3.summary()

import seaborn as sns
sns.regplot(X_temp, y)

sns.heatmap(X_temp.corr().abs())

X.plot(figsize = (20,10))


drop_list = ['BDEBT',
             'FGRECPT',
             'AIC4W',
             'CBYAAA']
X.drop(drop_list, axis = 1).plot(figsize = (20,10))



X['CBYAAA'].plot(figsize = (20,10))


##################
# index prediction

Y = (1 + y).cumprod()
Y.plot()
y.plot()

SPY

'SPY' in X.columns

result_ind = sm.OLS(Y,X).fit()
result_ind.summary()

ypred = result_ind.predict(X)

rpred = ypred.diff()

result_pred = sm.OLS(y,rpred,missing = 'drop').fit()
result_pred.summary()













































































































































