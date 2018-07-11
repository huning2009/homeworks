# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:29:44 2018

@author: lenovo
"""
path='D:/study/data_temp/factor_mat/'   #文件地址

def read_factor(path):
    import h5py
    import pandas as pd
    import numpy as np
        
    factor1=h5py.File(path+'factor_set.mat')
    factor=pd.DataFrame(factor1['factor_set'][:].T,columns=['beta','momentum','size','earnings','volatility'\
                 ,'growth','value','leverage','liquidity','non-linear','reversal'])
    date_temp=pd.read_csv(path+'date.csv')
    date=pd.DataFrame(np.tile(date_temp,(3628,1)),columns=['date'])
    stockid_temp=pd.read_csv(path+'stockid.csv')
    stockid=pd.DataFrame(np.tile(stockid_temp.T,(2777,1)).T.reshape(2777*3628,1),columns=['stockid'])
    date_stock=pd.merge(date,stockid,left_index=True,right_index=True)
    factor_set=pd.merge(date_stock,factor,left_index=True,right_index=True)
    
    return factor_set

















































