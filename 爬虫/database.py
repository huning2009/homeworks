import pandas as pd
import pymysql
import os
#���ݿ����ƺ�����
name = 'root'
password = 'dwy96649958866'
#�����������ݿ�����
db=pymysql.connect('localhost', name, password, charset='utf8')
cursor = db.cursor()
#�������ݿ�db_stock
sq1 = "create database db_stock"
cursor.execute(sq1)
#ѡ��ʹ�õ�ǰ���ݿ�
sq2 = "use db_stock;"
cursor.execute(sq2)
filepath="C:\\Users\\deng\\Desktop\\"
#��ȡ�����ļ��б�
fileList = os.listdir(filepath)
#���ζ�ÿ�������ļ����д洢
for fileName in fileList:
    data = pd.read_csv(filepath+fileName, encoding="gbk")
    #�������ݱ�
    sq3 = """create table stock
       (code VARCHAR(10), name VARCHAR(25),price float,\
                       change float��jinge float, jingzhangbi float)"""
    cursor.execute(sq3)
    #������ȡ����ÿ�����ݣ����δ洢
    length = len(data)
    for i in range(0, length):
        record = tuple(data.loc[i])
        try:
            sq4 = "insert into movie(code, name, price, change, jinge, jingzhangbi)\
           values (%s,%s,%s,%s,%s,%s)" % record
            #��ȡ�ı������ݺ��ң�����ȱʧֵ��Nnone��none�ȣ��������ݿ���Ҫ����ɿ�ֵ
            sq4 = sq4.replace('nan','null').replace('None','null').replace('none','null') 
            cursor.execute(sq4)
        except:
            #������ϲ�����̳��������������ݼ�¼���������½���
            break
#�ر��α꣬�ύ���ر����ݿ�����
cursor.close()
db.commit()
db.close()
