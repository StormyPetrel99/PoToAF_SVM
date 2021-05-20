import wfdb
import pywt
import matplotlib.pyplot as plt
import torch
import seaborn
import datetime
import sklearn
import numpy as np
from scipy import interpolate
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from lempel_ziv_complexity import lempel_ziv_complexity


PATH = 'af-termination-challenge-database-1.0.0/learning-set/'
ClassSet = ['n','s','t']
NumberSet = ['01','02','03','04','05','06','07','08','09','10']

PATH2 = 'af-termination-challenge-database-1.0.0\test-set-a'

PATH3='MIT-BIH-atrial-fibrillation-database-1.0.0/'
afdb_num=['04015','04043','04048','04126','04746',
          '04908','04936','05091','05121','05261',
          '06426','06453','06995','07162','07859',
          '07879','07910','08215','08219','08378',
          '08405','08434','08455']

#s = '1001111011000010' 
#print(s)
#print(lempel_ziv_complexity(s))

y_data=[]
vector1=[0]*30
vector2=[0]*30
j=0
for type in ClassSet:
    for number in NumberSet:
        file_path=PATH+type+number
        annotation = wfdb.rdann(file_path, 'qrs')
        samdata=annotation.sample
        tmp=[0]*(len(samdata)-1)
        for i in range(1,len(samdata)):
            tmp[i-1] = samdata[i]-samdata[i-1]
            tmp=np.array(tmp)
            tmp1=np.mat(tmp)
            tmp2=np.mean(tmp1)
            vector2[j]=tmp2
            SDRR=''
            for a in tmp:
                if(a>=tmp2):
                    SDRR=SDRR+'1'
                else:
                    SDRR=SDRR+'0'
        value=lempel_ziv_complexity(SDRR)
        vector1[j]=value
        j=j+1
vector1=np.array(vector1)
vector1=vector1.reshape(-1,1)
vector2=np.array(vector2)
vector2=vector2.reshape(-1,1)

#n1t0
for i in range(20):
    y_data.append(1)
for i in range(10):
    y_data.append(0)
y_data=np.array(y_data).reshape(-1,1)

vector=np.hstack((vector1,vector2))
train_data=np.hstack((vector,y_data))
print(train_data.shape)
print(train_data)

x,y = np.split(train_data,(2,),axis=1)
x = x[:,:2]
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.7)

model = svm.SVC(C=0.8, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
model.fit(x_train, y_train.ravel())
y_pre = model.predict(x_train)
print("训练集精度=" + str(sklearn.metrics.accuracy_score(y_train,y_pre)*100)+"%")
y_pre = model.predict(x_test)
print("测试集精度=" + str(sklearn.metrics.accuracy_score(y_test,y_pre)*100)+"%")
















        






