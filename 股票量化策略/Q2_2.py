#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 23:32:42 2018

@author: kimmy
"""

import pandas as pd
import numpy as np
import scipy.io  

#question2
#采用一维排序(B/M，E/P)验证价值在A股市场是否可行?
#找出价值投资的最佳持有期?
#计算持有期内的累积收益?（不允许卖空）
# 输入：N只股票E/P(或B/P),的矩阵(M*N),月度数据
# 注：B/M 即为P/B 的倒数

# data input and preprocessing
def datainit():
    
    hatdata = scipy.io.loadmat('hat.mat')
    hat = hatdata['hat']
    tradingDateV = hatdata['Date'][:,0] 
    tickerUnivSR = np.concatenate(hatdata['Stk'][:,0])
    tickerNameUnivSR = np.concatenate(hatdata['Stk'][:,1])
    
    closedata = scipy.io.loadmat('closeprice.mat')
    closeM = closedata['ClosePrice']
    closeM = pd.DataFrame(closeM,index=tradingDateV,columns=tickerUnivSR)
    
    PB = scipy.io.loadmat('PB')
    PB = PB['PB']
    PB = pd.DataFrame(PB,index=tradingDateV,columns=tickerUnivSR)
        
    return closeM, hat, tickerUnivSR, tickerNameUnivSR, tradingDateV, PB

# data preprocessing
def limUpDown(closeM,tradingDateV,tickerUnivSR):
    
    ## caculate limit_up and limit_dowm price
    limUpM_hat, limDownM_hat = np.around(closeM.values * 1.05, decimals=2), np.around(closeM.values * 0.95, decimals=2)
    limUpM_noHat,limDownM_noHat = np.around(closeM.values * 1.1, decimals=2), np.around(closeM.values * 0.9, decimals=2)
    
    ## using hatFlag fill blanks in limit_up & limit_down matrixs
    limUpM = np.full((len(tradingDateV),len(tickerUnivSR)),np.nan)
    limDownM = np.full((len(tradingDateV),len(tickerUnivSR)),np.nan)
    
    hatFlag = (hat == 1)
    limUpM[hatFlag] = limUpM_hat[hatFlag]
    limUpM[~hatFlag] = limUpM_noHat[~hatFlag]
    limDownM[hatFlag] = limDownM_hat[hatFlag]
    limDownM[~hatFlag] = limDownM_noHat[~hatFlag]
    
    ## run loops writing an initial matrix
    iRows = len(tradingDateV)
    limUpDownFlagM = np.zeros((len(tradingDateV),len(tickerUnivSR)))
    
    for i in range(1,iRows):
        FlagV_limUp = (closeM.values[i,:] >= limUpM[i-1,:])
        FlagV_limDown = (closeM.values[i,:] <= limDownM[i-1,:])
        limUpDownFlagM[i,FlagV_limUp] = 1
        limUpDownFlagM[i,FlagV_limDown] = -1
    
    return limUpDownFlagM

closeM, hat, tickerUnivSR, tickerNameUnivSR, tradingDateV, PB = datainit()
limUpDownFlagM = limUpDown(closeM,tradingDateV,tickerUnivSR)

nDates = len(tradingDateV)
nStocks = len(tickerUnivSR)

# transfer daily data to month data
def transfer_datas(df):
    tradingMonthV = np.around(tradingDateV/100,decimals=0).astype('int')
    df_month = df.set_index(tradingMonthV)
    Flag = df_month.index.duplicated(keep = 'last')
    df_month = df_month[~Flag]
    return df_month

BM_month = transfer_datas(1/PB)
close_month = transfer_datas(closeM)
limUpDownFlagM = transfer_datas(pd.DataFrame(limUpDownFlagM)).values

# caculate returns
def caculate_ytm(BM_Vec,close_temp,K):
    
    tickerID = np.array(close_temp.columns)
    #计算持有期收益率
    retV = (close_temp.iloc[-1]/close_temp.iloc[0])-1
    
    #数组拼接
    resultMatrix = pd.DataFrame([BM_Vec,retV],index=['BM','ret_hold'],columns=tickerID)
    
    # 按BM值进行排序
    BM_sorted = resultMatrix.sort_values(by = resultMatrix.index[0], axis=1)
    # 取最小的10个为明星股，最大的10个为价值股
    Glamour = BM_sorted.iloc[:,:10]
    Value = BM_sorted.iloc[:,-10:]
    
    ret_Glamour = np.mean(Glamour.values[1])
    ret_Value = np.mean(Value.values[1])
    
    #以单利的方法转换成年华收益率
    ret_Glamour = ret_Glamour/K*12
    ret_Value = ret_Value/K*12
    
    return ret_Glamour,ret_Value

# caculate ret_matrix
    #为简化计算，这里取用测试的价值指标维度M为6
def caculate_ytmM(BM_month,close_month,K):
    
    BM_month = BM_month.iloc[:6]
    
    retGlamour = np.array([])
    retValue = np.array([])
    
    # filter limit_up
    FlaglimUpM = (limUpDownFlagM==1)
    FlaglimDownM = (limUpDownFlagM==-1)
    
    iMonth = len(BM_month)
    
    for i in range(iMonth):
        BM_Vec = BM_month.iloc[i]
        close_temp = close_month.iloc[i:i+1+K]
        
        # fliter：涨停不能买入 跌停不能卖出
        FlaglimUpV = FlaglimUpM[i+1]
        FlaglimDownV = FlaglimDownM[i+K+1]
        FlagV = FlaglimUpV + FlaglimDownV
        IdxV = np.where(~FlagV)[0]
        BM_Vec = BM_Vec[IdxV]
        close_temp = close_temp.iloc[:,IdxV]

        # fliter： 删除nan列
        BM_Vec = BM_Vec.dropna()
        close_temp = close_temp.dropna(axis = 1)
        tickerInsV = np.intersect1d(np.array(BM_Vec.index),np.array(close_temp.columns))
        BM_flag = np.in1d(np.array(BM_Vec.index),tickerInsV)
        close_flag = np.in1d(np.array(close_temp.columns),tickerInsV)
        BM_Vec = BM_Vec[BM_flag]
        close_temp = close_temp.iloc[:,close_flag]
        
        #计算明星股和价值股的年收益率
        ret_Glamour, ret_Value = caculate_ytm(BM_Vec,close_temp,K)
        
        #添加进array数组
        retGlamour = np.append(retGlamour,ret_Glamour)
        retValue = np.append(retValue,ret_Value)
        
    #计算平均值
    Glamour_mean = np.mean(retGlamour)
    Value_mean = np.mean(retValue)
    
    return Glamour_mean,Value_mean

#test process
#Glamour_mean,Value_mean = caculate_ytmM(BM_month,close_month,12)

def caculate_bestHoldingPeriod(BM_month,close_month):
    
    K_Glamour = np.array([])
    K_Value = np.array([])
    #测试持有期K在期限为以12个月为间隔的100个月的持有期变化
    for K in range(12,100,12):
        Glamour_mean,Value_mean = caculate_ytmM(BM_month,close_month,K)
        K_Glamour = np.append(K_Glamour,Glamour_mean)
        K_Value = np.append(K_Value,Value_mean)
        
    K_Glamour = pd.Series(K_Glamour,index=np.arange(12,100,12))
    K_Value = pd.Series(K_Value,index=np.arange(12,100,12))

    retCumK_Glamour = (1 + K_Glamour).cumprod()-1
    retCumK_Value = (1 + K_Value).cumprod()-1
    
    return K_Glamour,K_Value, retCumK_Glamour,retCumK_Value


K_Glamour,K_Value, retCumK_Glamour,retCumK_Value = caculate_bestHoldingPeriod(BM_month,close_month)
print (K_Glamour,K_Value, retCumK_Glamour,retCumK_Value)
