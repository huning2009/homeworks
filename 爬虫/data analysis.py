# -*- coding: utf-8 -*-


import pandas as pd
from matplotlib import pyplot as plt

Days = [1,3,5,10]
for i in range(4):
    days = Days[i]



    df = pd.read_csv('zjlx{}.csv'.format(days), usecols=[0,2,3,4,5])
    df = df.replace('-', 0)
    df = df.astype('float64')
    
    
    ins = df[df.jinge>0]
    
    inUp = ins[ins.change>0]
    
    inDown = ins[ins.change<0]
    
    out = df[df.jinge<0]
    
    outUp = out[out.change>0]
    
    outDown = out[out.change<0]
    
    nums = [len(inUp), len(inDown), len(outUp), len(outDown)]
    plt.figure(i)
    plt.pie(nums, labels=('inUp', 'inDwon', 'outUp', 'outDown'), shadow=True,autopct = '%3.1f%%',
            startangle = 90)
    plt.title('{}日资金流向与涨跌情况统计图'.format(days))
    print(len(inUp)/(len(inUp)+len(inDown)), len(inDown)/(len(inUp)+len(inDown)), 
          len(outUp)/(len(outUp)+len(outDown)), len(outDown)/(len(outUp)+len(outDown)))










































