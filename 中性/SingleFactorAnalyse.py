# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:12:28 2017

@author: pengfang
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss
#import seaborn as sns
import statsmodels.formula.api as sm
from itertools import product
import math
from DataLoad import *
#%%


class OutlierCleaner(object):

    def __init__(self,isplot=True):
        self.isplot=isplot
#        pass
    def __plothist(self,factor):
        plt.hist(factor.values, bins=12, color='b', alpha=.4)
    
    def __Nstd(self,thisGroup,N=3):
        #N倍标准差，异常值
        thisM=thisGroup.mean()
        thisS=thisGroup.std()
        cap=float(thisM+N*thisS)
        flo=float(thisM-N*thisS)
        thisGroupC=thisGroup.copy()
        thisGroupC[thisGroup>cap]=cap
        thisGroupC[thisGroup<flo]=flo
        return thisGroupC

    def __BoxPlot(self,thisGroup):
		#箱线图去异常值
        md = float(thisGroup.median())
        X=np.array(thisGroup)
        largerN = X[X>md]
        smallerN = X[X<md]
        mclst=[(((i-md) - (md-j))/(i - j)) for i, j in product(largerN.tolist(),smallerN.tolist())]
        mc = np.median(mclst)
        Q3 = np.percentile(thisGroup, 75)
        Q1 = np.percentile(thisGroup, 25)
        if mc<0 :
            cap = Q3 + 1.5*np.exp(3.5*mc)*(Q3-Q1)
            flo = Q1 - 1.5*np.exp(-4*mc)*(Q3-Q1)
        else:
            cap = Q3 + 1.5*np.exp(4*mc)*(Q3-Q1)
            flo = Q1 - 1.5*np.exp(-3.5*mc)*(Q3-Q1)
        thisGroupC=thisGroup.copy()
        thisGroupC[thisGroupC>cap]=cap
        thisGroupC[thisGroupC<flo]=flo
        return thisGroupC
    
    def __MAD(self,thisGroup):
        thisMed=thisGroup.median()
        thisMad=(thisGroup-thisMed).abs().median()
        cap=float(thisMed+3*1.4826*thisMad)
        flo=float(thisMed-3*1.4826*thisMad)
        thisGroupC=thisGroup.copy()
        thisGroupC[thisGroupC>cap]=cap
        thisGroupC[thisGroupC<flo]=flo
        return thisGroupC    
    
    def Nstd_sectional(self,factor):
        #N倍标准差，按选股日分组去异常值，截面数据
        newfactor=factor.groupby(by='date').transform(self.__Nstd,N=3)
#        newgroup=factor.groupby(by='date')
#        newfactor=newgroup.apply(self.__StandardErrorN)
        if self.isplot: self.__plothist(newfactor) 
        return newfactor
    
    def Nstd_pannel(self,factor):
        #N倍标准差，面板数据
        newfactor=self.__Nstd(factor,N=3)
        if self.isplot: self.__plothist(newfactor) 
        return 
    
    def MAD_sectional(self,factor):
        newfactor=factor.groupby(by='date').transform(self.__MAD)
        if self.isplot: self.__plothist(newfactor) 
        return newfactor

    def BoxPlot_sectional(self,factor):
        newfactor=factor.groupby(by='date').transform(self.__BoxPlot)
        if self.isplot: self.__plothist(newfactor) 
        return newfactor    
    
class Standardization(object):
    def __init__(self):
        pass
    
    def __ZScore(self,thisGroup):
        thisM=float(thisGroup.mean())
        thisS=float(thisGroup.std())
        thisGroupC=thisGroup.copy()
        thisGroupC=(thisGroupC-thisM)/thisS
#        print(thisGroupC)
        return thisGroupC
    
    def __QuantileChange(self,thisGroup):
        f=thisGroup.values
		#在进行行业quantile标准化，出现整个行业仅有一只股票的情形时，将该股票因子赋值为1-epsilon(因其为该行业最大)
        if len(f) == 1: 
            epsilon = 0.0001
            r = [1-epsilon]
        else:
            I=np.argsort(-f)
            g=f.copy()
            g[I]=np.arange(len(f))
            quantile=(g+1)/(len(f))
            r=ss.norm.ppf(quantile)
            r[quantile==1]=ss.norm.ppf((len(f)-0.5)/len(f)) #将inf用一个比max(r[quantile!=1])大一些的数代替
        return r

    def QuantileChange_Ind(self, factorGroup):
#        factorGroup['industry'] = indGroup.values
#        factorGroup=factorGroup.join(indGroup,how='left')
		#分行业进行quantile标准化
        indlst = [self.QuantileChange_sectional(groupedItem[1].drop('Industry',axis = 1)) for groupedItem in factorGroup.groupby(by='Industry')]
        inddf = pd.concat(indlst, ignore_index = False)
        inddf.reset_index(inplace=True)
        inddf2=inddf.sort_values(by=['date','code'])
        inddf2.set_index(['date','code'],inplace=True)
        return inddf2


    def ZScore_Ind(self, factorGroup):
#        factorGroup['industry'] = indGroup.values
        #分行业进行zscore标准化
        indlst = [self.ZScore_sectional(groupedItem[1].drop('Industry',axis = 1)) for groupedItem in factorGroup.groupby(by='Industry')]
        inddf = pd.concat(indlst, ignore_index = False)
        inddf.reset_index(inplace=True)
        inddf2=inddf.sort_values(by=['date','code'])
        inddf2.set_index(['date','code'],inplace=True)
        return inddf2

    def ZScore_sectional(self,factor):
        newfactor=factor.groupby(by='date').transform(self.__ZScore)
        return newfactor
    
    def QuantileChange_sectional(self,factor):
        newfactor=factor.groupby(by='date').transform(self.__QuantileChange)
        return newfactor

class Orthogonalized(object):
    #正交函数
    def __init__(self):
        pass 

    def get(self,data,Y_columns,X_columns,Dummy_columns=[],method='OLS'):
        #data原始数据
        #Dummy_columns需要包含在X_columns里面
        if type(Y_columns) is not list: Y_columns=[Y_columns]
        if type(X_columns) is not list: X_columns=[X_columns]
        if type(Dummy_columns) is not list: Dummy_columns=[Dummy_columns]
        self.yc,self.xc,self.dc=Y_columns,X_columns,Dummy_columns
       
        assert ~data.isnull().any().any(),'含nan'
#            data=data.dropna(0)
        
        Resid=[]
        for key, group in data.groupby(by='date'):
                tempResid=self.OLS(group)
                Resid.append(tempResid)
        Resid=pd.concat(Resid)
        return Resid

    def OLS(self,data):
        y=data.loc[:,self.yc]
        x=data.loc[:, self.xc]
        if len(self.dc)>0:
            x=pd.get_dummies(x,columns=self.dc)
#        x=sm.add_constant(x)
        results = sm.OLS(y,x).fit()
        r=pd.DataFrame(results.resid,columns=['factor'],index=data.index)
        return r

class Regression(object):
    #线性回归函数
    def __init__(self):
        pass
    
    def __summary(self,Para):
        #回归打印总结表格
        t = Para['t'].mean()
        mu = Para['f'].mean()*100
        pct = float((Para['t']>0).sum()/len(Para['t']))*100
        abs_t = Para['t'].abs().mean()
        x=np.array([t,mu,pct,abs_t])
        result = pd.DataFrame({'Value':np.round(x,2),u'指标名称': [u'因子收益序列t值',u'因子收益均值 % ',u't>0比例 % ',u'abs(t)均值']})
        result = result.set_index(['指标名称'])
        print(result)
	
    def __draw(self,Para):
        #回归画图函数
        plt.rcParams['font.sans-serif'] = ['SimHei'] #定义中文字体
        plt.figure(1)        
        fig, axes = plt.subplots(1,figsize=(10, 6))
        Para['f'].hist(color = 'b', alpha=0.6, bins =10)
        plt.ylabel('Frequency')
        plt.legend(['f value'], loc='upper right',fontsize=15)
        plt.title(u'因子系数f值分布',fontsize=15)
        plt.grid(True,color = 'k', alpha = 0.2)

        lenX=len(Para)
        lable=np.arange(0,lenX,12)
        
        plt.figure(2)
        fig2, axes2 = plt.subplots(1,figsize=(10, 6))
        Para['f'].plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['f value'], loc='upper left',fontsize=15)
        xticklabels = Para.index[lable].strftime('%Y-%m')
        axes2.set_xticks(lable)
        axes2.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True,color = 'k', alpha = 0.2)
        plt.title(u'因子系数f值时间序列',fontsize=15)

        plt.figure(3)
        fig3, axes3 = plt.subplots(1,figsize=(10, 6))
        Para['t'].abs().plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['abs_tstat'], loc='upper right',fontsize=15)
        xticklabels = Para.index[lable].strftime('%Y-%m')
        axes3.set_xticks(lable)
        axes3.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True,color = 'k', alpha = 0.2)
        plt.title(u'回归t值得绝对值',fontsize=15)
        plt.show()
        
    def OLS(self,data):
        y=data.loc[:,self.retName]
        x=data.loc[:,self.factorName]
        results = sm.OLS(y,x).fit()
        f=results.params.iloc[0]
        t=results.tvalues.iloc[0]
        time=data.index[0][1]
        r=[time,f,t]
        return r
    
    def get(self,data,retName='futureRet',factorName='factor',method='OLS',isSummary=True,isPlot=True):
        data=data.loc[:,[retName,factorName]]
        assert ~(data.isnull().any().any()),'含nan报错'
        self.retName,self.factorName=retName,factorName
        Para=[] #记录检验值
        for key, group in data.groupby(by='date'):
            if method == 'OLS':
                tempPara=self.OLS(group)
            else:
                raise
            Para.append(tempPara)

                
        Para=pd.DataFrame(data=Para,columns=['time','f','t']).set_index(['time'])

        if isSummary:
            self.__summary(Para)
        if isPlot:
            self.__draw(Para)
            
        return Para

class IC(object):
    #IC计算
    def __init__(self):
        pass
    
    def __summary(self,ic,t):
        #IC打印总结表格
        lenIC=len(ic)
        mu =float(ic.rankIC.mean())
        std = float(ic.rankIC.std())
        pct = float((ic.rankIC>0).sum()/lenIC)*100
        IR = float(mu/std)
        meanT=float(ic.t.abs().mean())
        tt=sum(ic.t.abs()>t)/lenIC
        t_pos=sum((ic.rankIC>0) & (ic.t.abs()>t))/sum(ic.rankIC>0)*100
        t_neg=sum((ic.rankIC<0) & (ic.t.abs()>t))/sum(ic.rankIC<0)*100
        x=np.array([mu*100,std,pct,IR,meanT,tt,t_pos,t_neg])
        result = pd.DataFrame({'Value':np.round(x,2),'Name': ['IC_mean','IC_std','IC_pos_per', 'IR','t_mean','sign_all','sign_pos','sign_neg']})
        result = result.set_index('Name')
        return result

    def __draw(self,rankIC):
        #IC画图
        lenX=len(rankIC)
        lable=np.arange(0,lenX,12)
        fig, axes = plt.subplots(1,figsize=(10, 6))
        rankIC['rankIC'].plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['rankIC'], loc='upper right',fontsize=15)
        x = 1
        y = max(rankIC['rankIC'])
        plt.text(x,y, r'$\mu_{IC}=$'+str(round(rankIC['rankIC'].mean()*100,2))+'(%)',color='r',fontsize=15)
        xticklabels = rankIC.index[lable].strftime('%Y-%m')
        xtick = lable
        axes.set_xticks(xtick)
        axes.set_xticklabels(xticklabels, rotation=30,fontsize=15)
        plt.grid(True,color = 'k', alpha = 0.2)
        plt.show()

    def rankIC(self,data,retName='futureRet',factorName='factor',t=1.96,isSummary=True,isPlot=True):
        data=data.loc[:,[retName,factorName]]
        assert ~(data.isnull().any().any()),'含nan报错'
        ic=data.groupby(by='date').agg(lambda x:np.array(ss.spearmanr(x)))
        ic.columns=['rankIC','pValue']
        rs=ic.loc[:,'rankIC'].values
        ic.loc[:,'t'] = rs * np.sqrt((len(data)/len(rs)-2) / ((rs+1.0)*(1.0-rs)))

        #最后一期收益率都是0，因此应该抹去
        ic=ic.iloc[:-1]
        summary=self.__summary(ic,t)
        if isSummary:
            print(summary)
        if isPlot:
            self.__draw(ic)
        return ic,summary

class SpiltGroup(object):
    #分组
    def __init__(self):
        pass
    def __GroupInd(self,lenG):
        #分组的序号
        if lenG==1:
            GroupInd3 = np.array(self.GroupNum*[[0,1]])
        elif lenG==2:
            GroupInd3 = np.array(math.floor(float(self.GroupNum)/2.)*[[0,1]]+(self.GroupNum-math.floor(float(self.GroupNum)/2.))*[[1,2]])
        elif lenG<self.GroupNum:
            GroupInd=np.arange(lenG)
            y=self.GroupNum-lenG
            GroupInd2=[[i,i+1] for i in GroupInd]
            GroupInd2.extend(y*[GroupInd2[-1]])
            if len(GroupInd2)!=self.GroupNum: raise
            GroupInd3=np.array(GroupInd2)
        else:
            GroupInd=np.arange(0,lenG,np.floor(lenG/self.GroupNum))
            GroupInd2=[[GroupInd[i],(GroupInd[i+1] if (i+1)<=len(GroupInd)-1 else lenG)] for i in np.arange(self.GroupNum)] 
            GroupInd3=np.array(GroupInd2)
            GroupInd3[self.GroupNum-1,1]=lenG
        GroupInd3=GroupInd3.astype(int)
        if len(GroupInd3)!=self.GroupNum: raise
        return GroupInd3
    
    def __Group_Simple_AVG(self,data):
        #不考虑行业，简单的平均分组
        #按照主要因子排序
        data=data.sort_values(by=self.main,ascending = not(self.isDesc),inplace = False) 
        #分组
        GroupInd=self.__GroupInd(data.shape[0])

        #循环每一组
        for i in np.arange(self.GroupNum):
            #分组序号
            indexloc=data.index[GroupInd[i,0]:GroupInd[i,1]]

            #组号
            data.loc[indexloc,'group']=i
            #分配权重
            if self.WeightType == 'Simple_AVG':
                data.loc[indexloc,'weight']=1.00/len(indexloc)
            elif self.WeightType == 'MarketValue':
                #市值加权，输入的data应该有MV列
                data.loc[indexloc,'weight']=(data.loc[indexloc,'marketValue'].values)/data.loc[indexloc,'marketValue'].sum()
            else:
                raise #注意，简单分组里面不能用大行业

        return data
        
    def __Group_Industry_AVG(self,thisdate,data):
        #按照指数内的行业数量平均分组，当行业内的股票数量小于分组的时候，先顾头尾，
        #data必须包含Industry和MV列和WEIGHT列
        if self.isIndustryNeutral is True:
            uniDataIndustry=pd.DataFrame(pd.unique(data['industry']),columns=['industry'])
            industryWeight=self.__Weight_IndustryNeutralWeight(uniDataIndustry,thisdate)
            data=pd.merge(data,industryWeight,on='industry',how='left')
            if data['industryWeight'].isnull().any().any(): raise 
        else:
            data['industryWeight']=1
            
        data['denominator']=np.nan #权重分母
        #分组的组内list初始化
        groupList=[[] for i in np.arange(self.GroupNum)]
        #先按照行业分组，计算权重，并放入不同的组内
        for key, group in data.groupby(by='industry'):
            tempdata=group.sort_values(by=self.main,ascending = not(self.isDesc),inplace = False).copy() #按照主要因子排序
            GroupInd=self.__GroupInd(tempdata.shape[0])#一个行业五组分组序号
            for i in np.arange(self.GroupNum): #行业内分组
                tempdata2=tempdata.iloc[GroupInd[i,0]:GroupInd[i,1],:].copy()
                #行业内权重分母赋值
                if self.WeightType == 'Simple_AVG':
                    tempdata2.loc[:,'denominator']=len(tempdata2)
                elif self.WeightType == 'MarketValue':
                    tempdata2.loc[:,'denominator']=sum(tempdata2.loc[:,'marketValue'].values)
                #标明组号
                tempdata2.loc[:,'group']=i
                groupList[i].append(tempdata2)
        
        #整理分好组的部分
        groupListDF=[]
        for i in np.arange(self.GroupNum):
            thisDF=pd.concat(groupList[i])
            #计算组内权重
            if self.WeightType == 'Simple_AVG':
                thisDF.loc[:,'weight'] = 1/thisDF.loc[:,'denominator'].values*thisDF.loc[:,'industryWeight'].values
            elif self.WeightType == 'MarketValue':
                thisDF.loc[:,'weight'] = thisDF.loc[:,'marketValue']/thisDF.loc[:,'denominator'].values*thisDF.loc[:,'industryWeight'].values
            
            groupListDF.append(thisDF)

        result=pd.concat(groupListDF)
        return result

    def __Weight_IndustryNeutralWeight(self,uniDataIndustry,thisdate):
        #前提是，分组按行业来分，大行业的权重，行业内部可以是简单平均或者流通市值分组
        #注意，由于拿到的指数行业是延迟的，所以可能出现两种特殊情况，即：
        #1.指数内有某行业，但组中没有，此时去掉这些行业，重新计算每个行业的权重。注意，这里如果被抛弃的行业权重超过5%，将停止检查
        #2.组内有某行业，但指数中没有，那么把这个行业纳入进来，但是权重为0
        uniDataIndustry.loc[:,'value']=1
        if self.IndexIndustryWeight is None: raise
        thisIndexIndustryWeight=self.IndexIndustryWeight.loc[self.IndexIndustryWeight.loc[:,'date']==thisdate,['industry','industryWeight']]
#        IndexIndustry=thisIndexIndustryWeight.loc[:,'Industry']
        mergedata=pd.merge(thisIndexIndustryWeight,uniDataIndustry,how='outer')
        if mergedata.isnull().any().any():
            #当日行业权重为空，或者行业多出来
#        if (len(mergedata)!=len(thisIndexIndustryWeight)) | (len(uniDataIndustry)!=len(thisIndexIndustryWeight)):
#            raise #这段没实际应用过
            mergedata.loc[mergedata['industryWeight'].isnull(),['industryWeight']]=0
            mergedata.dropna(0,how='any',inplace=True)
            if sum(mergedata['industryWeight'])<=0.95: raise
            mergedata.loc[:,'industryWeight']=mergedata.loc[:,'industryWeight'].values/mergedata.loc[:,'industryWeight'].sum()
        
        mergedata=mergedata.loc[:,['industry','industryWeight']]
        if mergedata.loc[:,'industryWeight'].isnull().any(): raise
        return mergedata

    def get(self,data,indexCode,main='factor',isDesc=True,minor=None,needRet=False,GroupNum=5,SpiltGroupType='Simple_AVG',\
            WeightType='Simple_AVG',isIndustryNeutral=False,isplot=True):
        #data为数据源，main为分组中的主要因子，用这个因子来排序和分组
        #minor为次要因子，用来探索次要因子相对于主要因子的单调性，isDesc为因子方向，默认是【降序】排列
        #若minor次要因子为超额收益率，那么可以使needRet为True，从而计算收益净值曲线
        #GroupNum为分组数，默认为5，也可以设置为10；
        #SpiltGroupType为分组方式选择，默认为简单平均分组，可选择，按行业数量平均分配到各组
        #WeightType为分组后的权重分配。默认为简单平均分配权重
        #isIndustryNeutral为是否使用行业中性分配大行业的权重，注意，行业中性仅在使用行业分组中
        #使用行业中性权重时，需要将IndexIndustryWeight赋值
        #isplot控制是否画图，默认为True画图
        
        dataOri=data.copy()
        portfolio_list=data.reset_index().loc[:,['date','code']]
        selectDate=pd.to_datetime(pd.unique(portfolio_list.loc[:,'date']))
        
        #加入行业列
        Industry=DataLoad().loadFactor('Industry',portfolio_list)
        Industry=Industry.rename(columns={'factor':'industry'})
        data=data.join(Industry,how='left',rsuffix='_Ind')
        
        if data.isnull().any().any(): raise
        
        print('分组计算开始')
        if main=='factor':
            print('主要因子默认为factor')
        if minor is None:
            print('次要因子或收益率需注明')
            raise(IOError)
        if isIndustryNeutral:
#            print('权重中性化:导入指数行业权重')
            indexIndustryWeight=IndexWeightPort().getIndexIndustryWeight_Monthly(indexCode,selectDate)

        if WeightType=='MarketValue': 
#            print('流通市值加权，导入流通市值')
            MV=DataLoad().loadFactor('StockMarketValue_',portfolio_list)
            data=data.join(MV,how='left',rsuffix='_MV')
            data=data.rename(columns={'factor_NV':'marketValue'})
            if data.isnull().any().any(): raise
        #赋值
        self.isDesc,self.GroupNum,self.main,self.minor,self.WeightType,self.needRet,self.isIndustryNeutral,\
        self.IndexIndustryWeight=\
        isDesc,GroupNum,main,minor,WeightType,needRet,isIndustryNeutral,indexIndustryWeight
        
        data.reset_index(inplace=True)
        for i in ['group','weight','cumret']: #组号，权重，收益率
            data[i]=np.nan #组号
        
        
        #按日期循环
        newdata=[]
        for key, group in data.groupby(by='date'):
            if SpiltGroupType == 'Simple_AVG':
                result=self.__Group_Simple_AVG(group)
            elif SpiltGroupType == 'Industry_AVG':
                result=self.__Group_Industry_AVG(key,group)
            newdata.append(result)
        #分组后新数据拼接
        newdata=pd.concat(newdata)
        newdata.reset_index(drop=True,inplace=True)
        newdata=newdata.loc[:,['date','code','group','weight']]
        
        data=pd.merge(newdata,dataOri.reset_index(),how='left',on=['code','date'])
        meanMain=data.groupby(['group','date'])[self.main].mean()
        meanMain=meanMain.unstack('group')
        
        meanMinor=data.groupby(['group','date'])[self.minor].mean()
        meanMinor=meanMinor.unstack('group')
        
        turnoverRate=self.turnoverRate(newdata)
        result={'groupData':newdata,'mono':meanMinor,'turnoverRate':turnoverRate}
        
        #单调性画图
        if isplot:
            fig=plt.figure()
            ax1=fig.add_subplot(2,2,1)
            ax21=fig.add_subplot(2,2,2)
            ax3=fig.add_subplot(2,2,3)
            ax4=fig.add_subplot(2,2,4)
            plt.grid(True) #添加网格
            plt.ion()  #interactive mode on
            
            ax1.title.set_text('Stabilization of Main Factor ')
            ax21.title.set_text('Monotonicity of Minor Factor')
            ax3.title.set_text('Turnover Rate')
            ax4.title.set_text('Net Value of Different Group ')
            
            for i in np.arange(GroupNum): ax1.plot(np.arange(len(meanMain)), meanMain[i].T.values)#,'',label="Group"+str(i+1))
#            ax1.legend(loc = 'upper left', fontsize=15)
            
            barwidth=0.35
            ax21.bar(np.arange(GroupNum), meanMain.mean(axis=0),barwidth, color='b', alpha=0.7)
            ax22 = ax21.twinx()  # this is the important function
            ax22.bar(np.arange(GroupNum)+barwidth, meanMinor.mean(axis=0), barwidth, color='r', alpha=0.7)
            self.__align_yaxis(ax21, ax22,)
#            plt.legend([r'$\mu_{group}$'], loc='upper right',fontsize=15)
            
            ax3.bar(np.arange(GroupNum), turnoverRate.mean(axis=0))
            ax3.legend(loc='upper left',fontsize=15)

            
        if self.needRet:
            data.loc[:,'w_ret']=(1+data.loc[:,self.minor].values)*data.weight
            sumRet=data.groupby(['group','date']).w_ret.sum()
            sumRet=sumRet.unstack('group')
            
            
            netV=sumRet.cumprod(axis=0) #粗略分组净值
            #需要画图:均值--bar，sumRet--plot彩色
            if isplot:
                plotData = pd.DataFrame(netV,index = np.unique(data['date']))
                for i in np.arange(GroupNum): ax4.plot(np.arange(len(plotData)), plotData[i].T.values,'C'+str(int(i)),label="Group"+str(i+1))
                tick=np.arange(0,len(plotData),12)
                label=plotData.index[tick].strftime('%Y-%m')
                label=label.tolist()
                ax4.legend(loc='upper left',fontsize=15)
                ax4.set_xticks(tick)
                ax4.set_xticklabels(label,rotation=40)

                ax4.set_xlabel('Date')
                ax4.set_ylabel('Net Value')
                plt.show()

        return result
    
    def __align_yaxis(self,ax1, ax2):
        #画图专用
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = (ax1, ax2)
        extrema = [ax.get_ylim() for ax in axes]
        tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]
        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]
    
        b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
        t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
        axes[0].set_ylim(extrema[0][0], b_new_t)
        axes[1].set_ylim(t_new_b, extrema[1][1])
        
    def turnoverRate(self,groupData):
        rateList=[]
        for key1,data1 in groupData.groupby('group'):
            lastCode=[]
            for key2,data2 in data1.groupby('date'):
                nowCode=data2.code.tolist()
                if len(lastCode)>0:            
                    rate=len(set(lastCode)-set(nowCode))/len(lastCode)
                    rateList.append([key1,key2,rate])
                else:
                    rateList.append([key1,key2,np.nan])
                lastCode=nowCode.copy()
        
        rateList=pd.DataFrame(rateList,columns=['group','date','rate'])
        rateList2=rateList.set_index(['date','group'])
        rateList3=rateList2.unstack('group')
        return rateList3
        
    def GroupToPortfolio(self,data,GroupWeight=[1]):
        #GroupWeight是组比重的list
        #如果GroupWeight没给参数，那就是1份第一组
        if type(GroupWeight) != list: raise
        if (len(GroupWeight) == 1) & (GroupWeight[0] != 1): raise
        Portfolio=[]
        data.reset_index(inplace=True)
        for i in np.arange(len(GroupWeight)):
            thisPortfolio=data.loc[data['group']==i,['code','date','weight']].copy()
            thisPortfolio.loc[:,'weight']=thisPortfolio.loc[:,'weight']*GroupWeight[i]
            Portfolio.append(thisPortfolio)
        Portfolio=pd.concat(Portfolio)
        if len(GroupWeight)>1:
            Portfolio=Portfolio.groupby(by=['code','date']).sum()
        Portfolio.reset_index(inplace=True)
        Portfolio=Portfolio.loc[Portfolio.weight != 0,['code','date','weight']]
        return Portfolio

    def PortfolioToExcel(self,Portfolio,names='Portfolio.xlsx'):
        nextDay=DatePort().getNextDay()
        Portfolio=pd.merge(Portfolio,nextDay,how='left',on=['date'])
        Portfolio=Portfolio.loc[:,['code','dateN','weight']]
        Portfolio=Portfolio.rename({'dateN':'date'})

        from ReadConf import ReadConf
        path = ReadConf().get('Offline','path')
        names=path+names
        if os.path.exists(names):
            os.remove(names)
        writer = pd.ExcelWriter(names)
        Portfolio.to_excel(excel_writer=writer,index=False,header=False)
        writer.save()