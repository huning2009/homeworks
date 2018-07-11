# -*- coding: utf-8 -*-
"""
Created on Fri Jun 1 22:11:45 2018

@author: kimmy/huyuman
"""

import scipy.io
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math

closepriceData = scipy.io.loadmat('ClosePriceADJ.mat')
closeprice = closepriceData['ClosePrice']
retIndex = scipy.io.loadmat('Ret000001.mat')
indexRetM = retIndex['Ret000001']
indexRetM = indexRetM[1:,:]

df_closeprice = pd.DataFrame(closeprice)
#删除带nan的股票列
df_closeprice = df_closeprice.dropna(axis=1)
stockRetM = df_closeprice.pct_change()
stockRetM = stockRetM.iloc[1:,:].values

def count_tracking_error(FR, IR):
    # FR/IR: 列向量或者Series
    return np.mean(math.sqrt(np.sum(np.square(FR - IR))))

# 计算预期收益率'
def count_excess_return(FR, IR):
    return np.mean(FR - IR)

# 计算最优化方程的值
def func(w, other_params):
    w = np.mat(w).T       # 列向量
    R = other_params['R']
    IR = other_params['IR']
    lamda = other_params['lamda']
    FR = R * w
    TE = count_tracking_error(FR, IR)
    ER = count_excess_return(FR,IR)
    target = len(IR) * lamda * TE**2 - (1-lamda) * ER
    return target

# 权重之和为1 
def weight_constraint(w):
    return np.sum(w) - 1.0

# 计算最优权重
def count_best_weight(indexRetM, stockRetM, lamda, weight_limit=None):
    R = np.mat(stockRetM)
    IR = np.mat(indexRetM).reshape(len(indexRetM),1)
    w0 = np.full(R.shape[1],1/R.shape[1])
    other_params = {'R': R, 'IR':IR,'lamda':lamda}
    cons = {'type':'eq', 'fun':weight_constraint}
    bou=((0,1),)*len(w0)
    model = minimize(func, w0, args=other_params, bounds=bou, method='SLSQP', constraints = cons)
    return model['x']
    
result = count_best_weight(indexRetM, stockRetM, 0.5, weight_limit=None)