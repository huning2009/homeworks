import pandas as pd
import pymysql
import os
#数据库名称和密码
name = 'root'
password = 'dwy96649958866'
#建立本地数据库连接
db=pymysql.connect('localhost', name, password, charset='utf8')
cursor = db.cursor()
#创建数据库db_stock
sq1 = "create database db_stock"
cursor.execute(sq1)
#选择使用当前数据库
sq2 = "use db_stock;"
cursor.execute(sq2)
filepath="C:\\Users\\deng\\Desktop\\"
#获取本地文件列表
fileList = os.listdir(filepath)
#依次对每个数据文件进行存储
for fileName in fileList:
    data = pd.read_csv(filepath+fileName, encoding="gbk")
    #创建数据表
    sq3 = """create table stock
       (code VARCHAR(10), name VARCHAR(25),price float,\
                       change float，jinge float, jingzhangbi float)"""
    cursor.execute(sq3)
    #迭代读取表中每行数据，依次存储
    length = len(data)
    for i in range(0, length):
        record = tuple(data.loc[i])
        try:
            sq4 = "insert into movie(code, name, price, change, jinge, jingzhangbi)\
           values (%s,%s,%s,%s,%s,%s)" % record
            #获取的表中数据很乱，包含缺失值、Nnone、none等，插入数据库需要处理成空值
            sq4 = sq4.replace('nan','null').replace('None','null').replace('none','null') 
            cursor.execute(sq4)
        except:
            #如果以上插入过程出错，跳过这条数据记录，继续往下进行
            break
#关闭游标，提交，关闭数据库连接
cursor.close()
db.commit()
db.close()
