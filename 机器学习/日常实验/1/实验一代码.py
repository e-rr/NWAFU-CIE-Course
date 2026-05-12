#!/usr/bin/env python
# coding: utf-8

# In[17]:


import scipy.io as io
ministdata = io.loadmat("./datasets/MNIST.mat")
# feature,label = data["X"],data["Y"]


# In[14]:


lungdata = io.loadmat("./datasets/MNIST.mat")
# lungdata


# In[15]:


Yaledata = io.loadmat("./datasets/MNIST.mat")
# Yaledata


# In[8]:


len(feature),len(label)


# In[38]:


# coding: utf-8
import numpy as np
import math
from sklearn import datasets
from collections import Counter
infinity = float(-2**31) 
def sigmodFormatrix(Xb,thetas):
    params = - Xb.dot(thetas)
    r = np.zeros(params.shape[0])#返回一个np数组
    for i in range(len(r)):
        r[i] = 1 /(1 + math.exp(params[i]))
    return r
 
def sigmodFormatrix2(Xb,thetas):
    params = - Xb.dot(thetas)
    r = np.zeros(params.shape[0])#返回一个np数组
    for i in range(len(r)):
        r[i] = 1 /(1 + math.exp(params[i]))
        if r[i] >=0.5:
            r[i] = 1
        else:
            r[i] = 0
    return r
def sigmod(Xi,thetas):
    params = - np.sum(Xi * thetas)
    r = 1 /(1 + math.exp(params))
    return r
 
class LinearLogsiticRegression(object):
    thetas = None
    m = 0
    #训练
    def fit(self,X,y,alpha = 0.01,accuracy = 0.00001):
        #插入第一列为1，构成xb矩阵
        self.thetas = np.full(X.shape[1]+1,0.5)
        self.m = X.shape[0]
        a = np.full((self.m,1),1)
        Xb = np.column_stack((a,X))
        dimension  = X.shape[1]+1
        #梯度下降迭代
        count = 1
        while True:
            oldJ = self.costFunc(Xb, y)
            #注意预测函数中使用的参数是未更新的
            c = sigmodFormatrix(Xb, self.thetas)-y
            for j in range(dimension):
                self.thetas[j] = self.thetas[j] -alpha * np.sum(c * Xb[:,j])
            newJ = self.costFunc(Xb, y)
            if newJ == oldJ or math.fabs(newJ - oldJ) < accuracy:
                print("代价函数迭代到最小值，退出！")
                print("收敛到:",newJ)
                break
            print("迭代第",count,"次!")
            print("代价函数上一次的差:",(newJ - oldJ))
            count += 1
 
    #预测
    def costFunc(self,Xb,y):
        sum = 0.0
        for i in range(self.m):
            yPre = sigmod(Xb[i,], self.thetas)
            #print("yPre:",yPre)
            if yPre ==1 or yPre == 0:
                return infinity
            sum += y[i]*math.log(yPre)+(1 - y[i])*math.log(1-yPre)
        return -1/self.m * sum
    def predict(self,X):
        a = np.full((len(X),1),1)
        Xb = np.column_stack((a,X))
        return sigmodFormatrix2(Xb, self.thetas)
    def score(self,X_test,y_test):
        y_predict = myLogstic.predict(X_test)
        re = (y_test==y_predict)
        re1 = Counter(re)
        a = 1 - re1[True] / (re1[True]+re1[False])
        return a
#if __name__=="main":
from sklearn.model_selection import train_test_split

# ministdata
minist_x,minist_y =  ministdata["X"],[int(str(i)[1]) for i in ministdata["Y"]]
X_train1,X_test1, y_train1, y_test1 = train_test_split(minist_x,minist_y)
myLogstic = LinearLogsiticRegression()    
myLogstic.fit(X_train1, y_train1)
y_predict1 = myLogstic.predict(X_test1)
# print("参数:",myLogstic.thetas)
 
print("minist测试数据准确度:",round(myLogstic.score(X_test1, y_test1),5)) 
print("minist训练数据准确度:",round(myLogstic.score(X_train1, y_train1),5)) 

# lungdata
lung_x,lung_y =  ministdata["X"],[int(str(i)[1]) for i in ministdata["Y"]]
X_train2,X_test2, y_train2, y_test2 = train_test_split(lung_x,lung_y)
myLogstic = LinearLogsiticRegression()    
myLogstic.fit(X_train2, y_train2)
y_predict = myLogstic.predict(X_test2)
# print("参数:",myLogstic.thetas)
print("lung测试数据准确度:",round(myLogstic.score(X_test2, y_test2),5)) 
print("lung训练数据准确度:",round(myLogstic.score(X_train2, y_train2),5)) 

# Yaledata
Yale_x,Yale_y =  ministdata["X"],[int(str(i)[1]) for i in ministdata["Y"]]
X_train3,X_test3, y_train3, y_test3 = train_test_split(Yale_x,Yale_y)
myLogstic = LinearLogsiticRegression()    
myLogstic.fit(X_train3, y_train3)
y_predict = myLogstic.predict(X_test3)
# print("参数:",myLogstic.thetas)
 
print("Yale测试数据准确度:",round(myLogstic.score(X_test3, y_test3),5)) 
print("Yale训练数据准确度:",round(myLogstic.score(X_train3, y_train3),5)) 


# In[ ]:





# In[ ]:




