import numpy as np
import pandas as pd
import datetime as dt
import os
from matplotlib import pyplot as plt

################################ Fundamental Functions #################################

def process_data(Ret,Fea):
    # transfer date to index
    Fea.index = pd.to_datetime(Fea['date'])
    Ret.index = pd.to_datetime(Ret['date'])
    del Fea['date'], Ret['date']
    
    # strip the spaces
    Ret.columns = [x.strip() for x in Ret.columns]
    Fea.columns = [x.strip() for x in Fea.columns]
    
    # calculate the log return:
    Ret_simple = Ret
    Ret = np.log(1 + Ret)
    return Ret,Fea,Ret_simple
    
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

def change_fre(oneFea):  
    Fre_change_nna = oneFea.copy().dropna()
    # set value in the middle of a quarter to nan
    Fre_change_nna[Fre_change_nna == Fre_change_nna.shift(1)] = np.nan        
    # use linear model to fit the missing value
    Fre_change = Fre_change_nna.copy().interpolate(method = 'linear') 
    return Fre_change

################################ Data Processing Functions #################################
def cal_diff(Fea):
        
    ## Fea1 = Fea with no difference
    Fea1_name = ['GDP','LIBOR12M']
    Fea1 = Fea[Fea1_name]
    # Amend frequency by linear filling
    Fea1['GDP'] = change_fre(Fea['GDP'])
    
        
    ## Fea2 = Fea with simple difference
    Fea2_name = ['PSRATE','CFNAI']
    Fea2 = Fea[Fea2_name]
    # Get the difference
    Fea2_diff = Fea2.diff()
    Fea2_diff.columns = [x + '_diff' for x in Fea2.columns]
    Fea2 = Fea2.join(Fea2_diff)

    ## Fea3 = Fea with log difference
    Fea3 = Fea.copy().drop(Fea1_name + Fea2_name,axis = 1)
    # Amend frequency by linear filling
    GPDI = np.log(Fea3['GPDI'])
    Fea3['GPDI'] =  np.exp(change_fre(GPDI))
    # Get the difference
    Fea3_diff = np.log(Fea3).diff()
    Fea3_diff.columns = [x + '_log_diff' for x in Fea3_diff.columns]
    Fea3 = Fea3.join(Fea3_diff)
    
    # Aggregate all
    Fea_all = Fea1.join(Fea2).join(Fea3)

    return Fea_all


def add_lags(Fea, shift_lag = 3+1):
    Fea_org = cal_diff(Fea)
    Fea_extended = shift_generate(Fea_org, shift_lag)
    
    return Fea_extended

################################ the Main Function #################################

def Data_extend(Fea, shift_lag = 1 + 1):
    return add_lags(Fea,shift_lag)



