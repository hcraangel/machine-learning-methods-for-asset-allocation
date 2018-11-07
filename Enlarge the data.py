import numpy as np
import pandas as pd
import datetime as dt
import os
from matplotlib import pyplot as plt

################################ Functions #################################
def shift_generate(df, shift_lag = 6+1):

    df_name = df.columns
    df_out = df

    for i in np.arange(1,shift_lag):
        df_shift = df.shift(i)
        df_shi_name = [(x+'_lag_'+str(i)) for x in df_name]
        df_shift.columns = df_shi_name
        df_out = df_out.join(df_shift)
        del df_shift
    
    return df_out

################################ Process data #################################

os.chdir('F:\Capstone\Github\machine-learning-methods-for-asset-allocation')

## Dataframe processe
Ret = pd.read_csv('returns.csv')
Fea = pd.read_csv('features.csv')

os.chdir('F:\Capstone\Capstone\Codes\Build the big data')


Fea.index = pd.to_datetime(Fea['date'])
Ret.index = pd.to_datetime(Ret['date'])
del Fea['date'], Ret['date']

# strip the spaces
Ret.columns = [x.strip() for x in Ret.columns]
Fea.columns = [x.strip() for x in Fea.columns]

all_time = Ret.index

################################ Process data #################################
Ret_log = np.log(1 + Ret)

shift_lag = 1 + 3
## Fea1 = Fea with no difference
Fea1_name = ['GDP','LIBOR12M']
Fea1 = Fea[Fea1_name]
# change frequency
GDP = Fea1['GDP']
GDP_nna = GDP.copy().dropna()
GDP_nna[GDP_nna == GDP_nna.shift(1)] = np.nan           # set value in the middle of a quarter to nan
GDP_fre = GDP_nna.copy().interpolate(method = 'linear') # use linear model to fit the missing value
Fea1['GDP'] = GDP_fre
# Add lag items
Fea1_out = shift_generate(Fea1, shift_lag)

del GDP, GDP_nna, GDP_fre, Fea1


## Fea2 = Fea with simple difference
Fea2_name = ['PSRATE','CFNAI']
Fea2 = Fea[Fea2_name]
# Get the difference
Fea2_diff = Fea2.diff()
Fea2_diff.columns = [x + '_diff' for x in Fea2.columns]
Fea2_mid = Fea2.join(Fea2_diff)
# Add lag items
Fea2_out = shift_generate(Fea2_mid, shift_lag)

del Fea2, Fea2_diff, Fea2_mid

## Fea3 = Fea with log difference
Fea3 = Fea.copy().drop(Fea1_name + Fea2_name,axis = 1)
# Amend frequency by linear filling
GPDI = Fea3['GPDI'].copy().dropna()
GPDI_log = np.log(GPDI)
GPDI_log[GPDI_log == GPDI_log.shift(1)] = np.nan
GPDI_log_temp = GPDI_log.copy().interpolate(method = 'linear')
Fea3['GPDI'] = np.exp(GPDI_log_temp)
# Get the difference
Fea3_diff = np.log(Fea3).diff()
Fea3_diff.columns = [x + '_log_diff' for x in Fea3_diff.columns]
Fea3_mid = Fea3.join(Fea3_diff)
# Add lag items
Fea3_out = shift_generate(Fea3_mid, shift_lag)

del Fea3, GPDI, GPDI_log, Fea3_diff, Fea3_mid, Fea1_name, Fea2_name, 
del GPDI_log_temp, shift_lag

# Aggregate all
Fea_all = Fea1_out.join(Fea2_out).join(Fea3_out)

############################## select periods #################################
## plot the ratio between sample number and variable numbers
t = Fea_all.shape[0]
na_check_0 = (1 - Fea_all.isna()).sum()             # available time points for every feature
na_check_1 = (1 - Fea_all.isna()).sum(axis = 1)     # available features for every time point

na_check_1_per = (t - np.arange(len(na_check_1)))/na_check_1    # sample size divides by the feature number
na_check_1.plot(figsize = (16.18,10), title = 'Sample Size/Feature Number')
na_check_1_per[na_check_1_per < 50].plot(figsize = (10,7))

## check the time periods
# After the Second World War
from pandas.tseries.offsets import MonthEnd, YearEnd
SWW = dt.datetime(1945,9,30) + MonthEnd(6)
#Oil_cirsis = dt.datetime(1973,10,30)
na_check_1_per[SWW]
na_check_1[SWW]
Fea_SWW = Fea_all[Fea_all.index >= SWW]
Ret_SWW = Ret[Ret.index >= SWW]


############################## Regression #################################
import statsmodels.api as sm
X = Fea_SWW.dropna(axis = 1)
X = sm.add_constant(X)
y = Ret_SWW['SPY']

Reg1 = sm.OLS(y,X).fit()
Reg1.summary()
        
#aa = str(Reg1.summary())
#f = open('aa.txt','w+')
#f.write(aa)
#f.close()

from sklearn.linear_model import LinearRegression as LR
X = Fea_SWW.dropna(axis = 1)
y = Ret_SWW['SPY']

Reg2 = LR().fit(X,y)

y_fit = pd.Series(Reg2.predict(X),index = y.index)

np.sqrt(np.mean((y_fit - y)**2))

yy = pd.concat([y,y_fit],axis = 1)


#yy.plot(figsize = (60,40))
#plt.savefig('return vs return.fit.png')

import statsmodels.tsa as stt

ARMA_model = stt.arima_model.ARMA(y,order = (1,0), freq = 'M')
ARMA_result = ARMA_model.fit()
ARMA_result.summary()
y_fit_ARMA = ARMA_model.predict(y[0:len(y)-1])
y_fit_ARMA = pd.Series(ARMA_model.predict(y),index = y.index)
y_fit_ARMA

X_AR = y.shift(1)
X_AR = sm.add_constant(X_AR)

result_AR = sm.OLS(y, X_AR, missing = 'drop').fit()
result_AR.summary()






































































































































































































Fea3['GPDI'] == Fea3['GPDI'].shift(1)

Fea3_log = np.log(Fea3).diff()

Fea3_log['GPDI'].plot(figsize = (20,10))

st_list = ['GDP', 'LIBOR12M', 'CFNAI','PSRATE']
Fea_st = np.log(1 + Fea[st_list])



Fea_shi = Fea.shift(1)
Fea_name = Fea.columns
Fea_shi_name = [''.join([x, '_lag1']) for x in Fea_name]
Fea_shi.columns = Fea_shi_name


Fea_all = Fea.join(Fea_shi)

diff_all = np.log(Fea) - np.log(Fea_shi)



np.sum((Fea<0))












































