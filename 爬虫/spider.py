# -*- coding: utf-8 -*-


import urllib
import re
import pandas as pd
import time

def scrap1():
    df = pd.DataFrame()
    for i in range(1,72):
        url=u'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx/JS.aspx?type=ct&st=(BalFlowMain)&sr=-1&p=' + str(i) + '&ps=50&js=var%20={pages:(pc),date:%222014-10-22%22,data:[(x)]}&token=894050c76af8597a853f5b408b759f5d&cmd=C._AB&sty=DCFFITA'
        page = urllib.request.urlopen(url)
        html = page.read().decode('utf-8', 'ignore')
        pattern1 = re.compile(u'\[(.*?)\]')
        needed = pattern1.findall(html)
        pattern2 = re.compile(u'"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)"')
        per_stock = pattern2.findall(needed[0])
        data = pd.DataFrame([list(i) for i in per_stock]).iloc[:, 1:7]
        data.columns = ['code', 'name', 'price', 'change', 'jinge', 'jingzhanbi']
        
        df = df.append(data)
    
    return(df)



def scrap3():
    df = pd.DataFrame()
    for i in range(1,72):
        url=u'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx/JS.aspx?type=ct&st=(BalFlowMainNet3)&sr=-1&p=' + str(i) + '&ps=50&js=var%20={pages:(pc),date:%222014-10-22%22,data:[(x)]}&token=894050c76af8597a853f5b408b759f5d&cmd=C._AB&sty=DCFFITA3'
        page = urllib.request.urlopen(url)
        html = page.read().decode('utf-8', 'ignore')
        pattern1 = re.compile(u'\[(.*?)\]')
        needed = pattern1.findall(html)
        pattern2 = re.compile(u'"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)"')
        per_stock = pattern2.findall(needed[0])
        data = pd.DataFrame([list(i) for i in per_stock]).iloc[:, 1:7]
        data.columns = ['code', 'name', 'price', 'change', 'jinge', 'jingzhanbi']
        
        df = df.append(data)
    
    return(df)



def scrap5():
    df = pd.DataFrame()
    for i in range(1,72):
        url=u'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx/JS.aspx?type=ct&st=(BalFlowMainNet5)&sr=-1&p=' + str(i) + '&ps=50&js=var%20={pages:(pc),date:%222014-10-22%22,data:[(x)]}&token=894050c76af8597a853f5b408b759f5d&cmd=C._AB&sty=DCFFITA5'
        page = urllib.request.urlopen(url)
        html = page.read().decode('utf-8', 'ignore')
        pattern1 = re.compile(u'\[(.*?)\]')
        needed = pattern1.findall(html)
        pattern2 = re.compile(u'"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)"')
        per_stock = pattern2.findall(needed[0])
        data = pd.DataFrame([list(i) for i in per_stock]).iloc[:, 1:7]
        data.columns = ['code', 'name', 'price', 'change', 'jinge', 'jingzhanbi']
        
        df = df.append(data)
    
    return(df)







def scrap10():
    df = pd.DataFrame()
    for i in range(1,72):
        url=u'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx/JS.aspx?type=ct&st=(BalFlowMainNet10)&sr=-1&p=' + str(i) + '&ps=50&js=var%20={pages:(pc),date:%222014-10-22%22,data:[(x)]}&token=894050c76af8597a853f5b408b759f5d&cmd=C._AB&sty=DCFFITA10'
        page = urllib.request.urlopen(url)
        html = page.read().decode('utf-8', 'ignore')
        pattern1 = re.compile(u'\[(.*?)\]')
        needed = pattern1.findall(html)
        pattern2 = re.compile(u'"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)"')
        per_stock = pattern2.findall(needed[0])
        data = pd.DataFrame([list(i) for i in per_stock]).iloc[:, 1:7]
        data.columns = ['code', 'name', 'price', 'change', 'jinge', 'jingzhanbi']
        
        df = df.append(data)
    
    return(df)


if __name__ == '__main__':
    start = time.clock()
    df1 = scrap1()
    df3 = scrap3()
    df5 = scrap5()
    df10 = scrap10()
    end = time.clock()
    print(end-start)
    df1.to_csv('zjlx1.csv', index=False)
    df3.to_csv('zjlx3.csv', index=False)
    df5.to_csv('zjlx5.csv', index=False)
    df10.to_csv('zjlx10.csv', index=False)














