# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:15:33 2018

@author: lenovo
"""
import pandas as pd
import numpy as np
from datetime import datetime

month_data=pd.read_csv('D:/study/data_temp/factor/month_data.csv');
year_data=pd.read_csv('D:/study/data_temp/factor/year_data.csv');
close_day=pd.read_csv('D:/study/data_temp/factor/close_day.csv',header=None);
ret_day=pd.read_csv('D:/study/data_temp/factor/ret_day.csv',header=None);
vol_day=pd.read_csv('D:/study/data_temp/factor/volume_day.csv',header=None);
adjFactor_day=pd.read_csv('D:/study/data_temp/factor/adjFactor_day.csv',header=None);
date_day=pd.read_csv('D:/study/data_temp/factor/date_day.csv');
code_day=pd.read_csv('D:/study/data_temp/factor/code_day.csv',encoding='gbk');


#去掉单引号
func=lambda x:x[1:len(x)-1]
month_data['code']=month_data['code'].apply(func)
month_data['date']=month_data['date'].apply(func)
year_data['code']=year_data['code'].apply(func)
year_data['date']=year_data['date'].apply(func)
date_day['date']=date_day['date'].apply(func)
code_day['code']=code_day['code'].apply(func)
code_day['name']=code_day['name'].apply(func)

#拼接day_data
day_data=pd.DataFrame(columns={'code','date','price','ret','vol','adjFactor','industry'},index=range(0,len(close_day)*len(close_day.columns)))
day_data['price']=pd.DataFrame(np.mat(close_day).T.reshape(len(close_day)*len(close_day.columns),1))
day_data['ret']=pd.DataFrame(np.mat(ret_day).T.reshape(len(ret_day)*len(ret_day.columns),1))
day_data['vol']=pd.DataFrame(np.mat(vol_day).T.reshape(len(vol_day)*len(vol_day.columns),1))
day_data['adjFactor']=pd.DataFrame(np.mat(adjFactor_day).T.reshape(len(vol_day)*len(vol_day.columns),1))

day_data['code']=pd.DataFrame(np.tile(np.mat(code_day.code),(len(close_day),1)).T.reshape(len(vol_day)*len(vol_day.columns),1))
day_data['industry']=pd.DataFrame(np.tile(np.mat(code_day.industry),(len(close_day),1)).T.reshape(len(vol_day)*len(vol_day.columns),1))

day_data['date']=pd.DataFrame(np.tile(np.mat(date_day.date).T,(len(close_day.columns),1)));


#winda
winda=pd.read_csv('D:/study/data_temp/factor/winda.csv');
day_data=pd.merge(day_data,winda,how='inner',on='date')    


#deal datetime
str_time=lambda x: pd.Timestamp(x)  
month_data.date=month_data.date.apply(str_time)
day_data.date=day_data.date.apply(str_time)
year_data.date=year_data.date.apply(str_time)


#calc month_ret
get_month=lambda x: x.month
get_year=lambda x: x.year
day_data['month']=day_data.date.apply(get_month)
temp1=day_data[['date','month']][day_data.code=='000001.SZ']
month_end=temp1['date'].iloc[np.where(np.array(temp1.month[0:len(temp1)-1])!=np.array(temp1.month[1:len(temp1)]))[0]]
price_month=day_data[day_data['date'].isin(month_end)].reset_index(drop=True)
price_month['year']=price_month.date.apply(get_year)
month_data['year']=month_data.date.apply(get_year)
month_data['month']=month_data.date.apply(get_month)
month_data=month_data.set_index(['year','month','code'])
price_month=price_month.set_index(['year','month','code'])
month_data=pd.merge(month_data,price_month[['price','adjFactor','price_bench']],how='left',left_index=True,right_index=True)
month_data=month_data.reset_index()


day_data=day_data.set_index(['code','date'])
month_data=month_data.set_index(['code','date'])
year_data=year_data.set_index(['code','date'])



























