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
