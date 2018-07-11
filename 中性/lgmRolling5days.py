# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgm
from sklearn import metrics
#from sklearn.model_selection import cross_val_score
import os
import h5py
import pickle

os.chdir('/Users/kimmy/signal_ensemble')

class lgm_trainer():
    
    def __init__(self):
        
        self.data, self.totalOrg, self.date, self.filenames, self.stk, self.columns = self._read_data()
        self.alambda = np.power(0.5,1/250)
        self.score = np.zeros((len(self.date), len(self.stk)))
        self.param = self.get_params()
        
        
    def _read_data(self):
        
        file1 = h5py.File('total.h5','r')
        data = file1['totalData'][:]
        totalOrg = file1['totalOrg'][:]
        file2 = open('pklData.pkl','rb')
        pklData = pickle.load(file2)
        date = pklData['x_dates']
        filenames = pklData['signal_names']
        stk = pklData['stock_names']
        file1.close()
        file2.close()

        column = np.append(['category','dateIndex'],filenames)
        data = pd.DataFrame(data,index=pklData['dataIndex'],columns=column)
        totalOrg = pd.DataFrame(totalOrg,index=pklData['orgIndex'],columns=column)
        
        return data, totalOrg, date, filenames, stk, column
        
    def get_params(self):

        param = {}
        # use softmax multi-class classification
        #param['application'] = 'xentropy'
        #param['boosting']='gbdt'
        param['objective'] = 'binary'
        param['learning_rate'] = 0.2
        param['max_depth'] = 5
        param['num_leaves'] = 25
        param['feature_fraction']=0.85
#        param['device'] = 'gpu'
#        param['gpu_device_id'] = 1
        #param['bagging_fraction'] = 0.9
        #param['bagging_freq'] = 4
        param['train_metric']=True
        param['metric']='binary_error'
        #param['metric']='binary_logloss'
        #param['metric']='xentropy'
        #param['metric']='xentlambda'
        #param['metric']='auc'
        #param['metric']='map'
        param['use_missing']=False
        
        return param
    
    def __data_clean(self):
        X = self.totalOrg.loc[:, self.filenames.tolist()]
        Y = self.totalOrg.loc[:, ['category']]
        dataIndex = self.totalOrg.loc[:, ['dateIndex']]
        Y_nonzerosIdx = np.nonzero(Y.values)
        Y = Y.iloc[Y_nonzerosIdx[0]]
        X = X.iloc[Y_nonzerosIdx[0]]
        dataIndex = dataIndex.iloc[Y_nonzerosIdx[0]]
        Yflag = (Y == -1)
        Y[Yflag] = 0
        return X,Y,dataIndex

    
    def lgm_training(self):
        X,Y,dataIndex = self.__data_clean()
        
        # accuracyList=[]
        # precisionList=[]
        # rankingpreciseList=[]
        # rankinglossList=[]
        
        for idate in range(500, len(self.date)-12, 5):
            if idate == 500:
                X_train = np.array(X.loc[self.date[idate - 500:idate]])
                Y_train = np.array(Y.loc[self.date[idate - 500:idate]]).T[0]
                #dataweights = np.power(self.alambda, idate - np.array(dataIndex.loc[self.date[idate - 500:idate]])).T[0]
            else:
                X_add = np.array(X.loc[self.date[idate - 5:idate]])
                Y_add = np.array(Y.loc[self.date[idate - 5:idate]]).T[0]
                X_train = np.concatenate((X_train, X_add), axis=0)
                Y_train = np.concatenate((Y_train, Y_add), axis=0)

            lgm_train = lgm.Dataset(X_train, label=Y_train)
            bst = lgm.train(self.param, lgm_train,num_boost_round=50)

            # get prediction
            for itestdate in range(idate+6, idate + 11):
                x = np.array(self.totalOrg.loc[self.date[itestdate],self.filenames.tolist()])
                #y = np.array(self.totalOrg.loc[self.date[itestdate],['category']])
                flagV = (np.sum(np.isnan(x),axis=1)>0)
                lgm_test = x[~flagV]
                pred_raw = bst.predict(lgm_test)
                pred = np.full(len(self.stk),np.nan)
                pred[~flagV] = pred_raw
                self.score[itestdate,:] = pred
               
                # y = y[~flagV]
                # x = x[~flagV]
                # y_nonzerosIdx = np.nonzero(np.array(y))
                # y = y[y_nonzerosIdx[0]]
                # x = x[y_nonzerosIdx[0]]
                # Yflag = (y == -1)
                # y[Yflag] = 0
                # Y_test = y.T[0].astype(int)
                # pred_raw = bst.predict(x)
                # ypred = ((pred_raw>0.5)*1).astype(int)
                # accuracy = metrics.accuracy_score(Y_test,ypred)
                # precision = metrics.precision_score(Y_test,ypred)
                # rankingPrecise = metrics.label_ranking_average_precision_score(np.mat(Y_test),np.mat(pred_raw))
                # label_ranking_loss = metrics.label_ranking_loss(np.mat(Y_test),np.mat(pred_raw))
               
                # output
                # accuracyList.append(accuracy)
                # precisionList.append(precision)
                # rankinglossList.append(label_ranking_loss)
                # rankingpreciseList.append(rankingPrecise)
                #
                print ('Date', self.date[itestdate], 'done' )
                # print ('ranking precision:%.4f'%rankingPrecise)
                # print ('ranking loss:%.4f'%label_ranking_loss)
                # print ('precision: %.4f'%precision)
                # print ('accuracy: %.4f'%accuracy)
    
        score = pd.DataFrame(self.score, columns=self.stk, index=self.date)
        score.to_csv('/Users/kimmy/signal_ensemble/lgmRollingAllScore_NA.csv')

        # precision = pd.DataFrame(precisionList,index=self.date[500:2650])
        # accuracy =  pd.DataFrame(accuracyList,index=self.date[500:2650])
        # rankingPrecise = pd.DataFrame(rankingpreciseList,index=self.date[500:2650])
        # rankingloss = pd.DataFrame(rankinglossList,index=self.date[500:2650])
        # precisionDatas = {'score':score,
        #                   'precision': precision,
        #                   'accuracy':accuracy,
        #                   'ranking precision': rankingPrecise,
        #                   'ranking loss': rankingloss}
        # pkls = open('C:/my code/LightGBMPrecision_1.pkl','wb')
        # pickle.dump(precisionDatas,pkls)
        # pkls.close()
        return


lightgbmAll = lgm_trainer()
lightgbmAll.lgm_training()

##ranking loss
# futRet = pd.read_csv('futRet.csv',index_col=0)
# y_true = futRet.iloc[505:1785,:].values
# flag = y_true==-1
# y_true[flag]=0
# y_pred = score.iloc[505:1785,:].values
# rankingPrecise = metrics.label_ranking_average_precision_score(y_true,y_pred)
# label_ranking_loss = metrics.label_ranking_loss(y_true,y_pred)
# print (rankingPrecise,label_ranking_loss)
# ###roc score
# futRet = pd.read_csv('futRet.csv',index_col=0)
# y_true = futRet.iloc[505:1785,:].unstack().to_frame().T.iloc[0].values
# flag = y_true==-1
# y_true[flag]=0
# y_pred = score.iloc[505:1785,:].unstack().to_frame().T.iloc[0].values
# roc_auc_score = metrics.roc_auc_score(y_true,y_pred)
# print (roc_auc_score)
