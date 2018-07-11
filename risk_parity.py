#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:49:03 2017

@author: songchengbin
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

"0.数据清洗"
##数据清洗
data=data.dropna(axis=1,how='any')
##选择样本
price=data.iloc[:,:4]

"1.计算历史风险和收益"
##计算对数收益率
ret=np.log([price.values.tolist()])
ret=ret[0]
ret=ret[1:]-ret[:-1]
ret=pd.DataFrame(ret,index=price.index[1:],columns=price.columns)
##转化为百分数
ret=ret*100
ret=round(ret,2)
##计算历史风险和收益
mean=np.mean(ret)
mean=round(mean,2)
cov=np.cov(ret.T)
cov=pd.DataFrame(cov,index=price.columns,columns=price.columns)
cov=round(cov,2)

"2.计算各资产的风险贡献"
#weight=np.ones(len(price.columns))/len(price.columns)
#MRC=np.mat(cov)*np.mat(weight).T
#TRC=np.multiply(MRC,np.mat(weight).T)
#TRC=pd.DataFrame(TRC,index=price.columns)

"3.优化求解，求最优权重"
def target(weight):
    a=np.multiply(np.mat(cov)*np.mat(weight).T,np.mat(weight).T)
    b=0
    for i in range(len(price.columns)):
        b=b+sum(np.array(a-a[i])**2)
    return b
def summ(weight):
    b=1-sum(weight)
    return b
weight=np.ones(len(price.columns))/len(price.columns)
constraint={'type':'eq','fun':summ}
weight_opt=scipy.optimize.minimize(target,weight,constraints=(constraint))
weight_opt=pd.DataFrame(weight_opt,index=price.columns)
weight_opt=weight_opt['x']

"4.验证风险贡献TRC"
MRC=np.mat(cov)*np.mat(weight_opt).T
TRC=np.multiply(MRC,np.mat(weight_opt).T)
TRC=pd.DataFrame(TRC,index=price.columns)

"5.画图风险平价和等权重"
def account(position):
    cumsumret=np.cumsum(ret/100)
    position=np.multiply(np.exp(cumsumret),np.mat(position))
    account=np.sum(position,axis=1)
    account=pd.DataFrame(account,index=ret.index)
    plt.plot(account)
    plt.show()
cash=1000000
position=np.ones(len(price.columns))*cash
position=np.multiply(np.mat(position),np.mat(weight_opt))
cumsumret=np.cumsum(ret/100)
position=np.multiply(np.exp(cumsumret),np.mat(position))
account=np.sum(position,axis=1)
account=pd.DataFrame(account,index=ret.index)
plt.plot(account)
plt.show()
position_average=np.ones(len(price.columns))/len(price.columns)*cash
