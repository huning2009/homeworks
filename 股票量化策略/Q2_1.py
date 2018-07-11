#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:46:41 2018

@author: kimmy/huyuman
"""

import pandas as pd
import numpy as np
import scipy.io  

# question 1
##采用日度数据，检验A股市场惯性策略在短期是否可行？
##找出不同(j,k)策略的收益情况？找出收益最高的(j,k)策略的收益？
##要求在J+1时刻排序，J+2时刻买入，J+K+2时刻卖出
##涨停不能买入，跌停不能卖出，不考虑交易费用
##投资组合持有股票个数自定，涨停不能买入的股票需要从后续的股票递补
    
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
    
    return closeM, hat, tickerUnivSR, tickerNameUnivSR, tradingDateV

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

closeM, hat, tickerUnivSR, tickerNameUnivSR, tradingDateV = datainit()
limUpDownFlagM = limUpDown(closeM,tradingDateV,tickerUnivSR)

nDates = len(tradingDateV)
nStocks = len(tickerUnivSR)

def caculate_ret(J,K,closeM_temp):

    tickerID= np.array(closeM_temp.columns)
    retJ = closeM_temp.pct_change(J) #caculate J period returns
    retK = closeM_temp.pct_change(K) #caculate K period returns
    retMatrix = pd.DataFrame([retJ.iloc[J],retK.iloc[K+2]],index=['ret_sorted','ret_hold'],columns=tickerID)
    
    # 取J+1时刻的累积收益率，进行排序
    ret_sorted = retMatrix.sort_values(by = retMatrix.index[0], axis=1)
    # 取时刻前10收益率的为赢家，后10收益率的为输家
    winner = ret_sorted.iloc[:,-10:]
    loser = ret_sorted.iloc[:,:10]

    #等权重赋予以上投资组合的10支股票，计算赢家收益率&输家收益率
    ret_winner = np.mean(winner.values[1])
    ret_loser = np.mean(loser.values[1])
    
    return ret_winner, ret_loser
 
def caculate_retCum(J,K,closeM):    

    # filter limit_up
    FlaglimUpM = (limUpDownFlagM==1)
    FlaglimDownM = (limUpDownFlagM==-1)
    
    retWinner = np.array([])
    retLoser = np.array([])
    iCols = nDates-J-K-2
    for i in range(iCols):
        closeM_temp = closeM.iloc[i:J+K+2+i,:]
        
        # fliter：涨停不能买入 跌停不能卖出
        FlaglimUpV = FlaglimUpM[i+J+2]
        FlaglimDownV = FlaglimDownM[i+J+K+2]
        FlagV = FlaglimUpV + FlaglimDownV
        IdxV = np.where(~FlagV)[0]
        closeM_temp = closeM_temp.iloc[:,IdxV]
        
        # fliter： 删除nan列
        closeM_temp = closeM_temp.dropna(axis = 1)
        
        ret_winner, ret_loser = caculate_ret(J,K,closeM_temp)
        
        retWinner = np.append(retWinner,ret_winner)
        retLoser = np.append(retLoser,ret_loser)
    
    retWinner_mean = np.mean(retWinner)
    retLoser_mean = np.mean(retLoser)
    
    return retWinner_mean, retLoser_mean

# test process
#ret_winner,ret_loser = caculate_retCum(100,100,closeM)
#print (ret_winner,ret_loser)

def caculate_retM(closeM):
    
    J = [60,120,240]
    K = [60,120,240]
    retWinner_Matrix = np.empty((3,3))
    retLoser_Matrix = np.empty((3,3))
    ret_Matrix = np.empty((3,3,2))
    for j in J:
        for k in K:
            ret_winner,ret_loser = caculate_retCum(j,k,closeM)
            retWinner_Matrix[J.index(j),K.index(k)]=ret_winner
            retLoser_Matrix[J.index(j),K.index(k)]=ret_loser
            ret_Matrix[J.index(j),K.index(k)] = np.array([ret_winner,ret_loser])
    
    return ret_Matrix,retWinner_Matrix,retLoser_Matrix

ret_Matrix,retWinner_Matrix,retLoser_Matrix = caculate_retM(closeM)
print (ret_Matrix,retWinner_Matrix,retLoser_Matrix)

