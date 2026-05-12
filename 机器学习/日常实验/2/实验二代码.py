#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[21]:


data = pd.read_csv("datasets/abalone.csv")


# In[22]:


data.head()


# In[23]:


data["Sex"].unique()


# In[24]:


data.loc[data['Sex']=='I','Sex'] = 0
data.loc[data['Sex']=='M','Sex'] = 1
data.loc[data['Sex']=='F','Sex'] = 2


# In[25]:


data['Sex'] = data['Sex'].astype('int')


# In[ ]:





# In[26]:


import numpy as np
import matplotlib.pyplot as plt
 
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
 
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    yMean = np.mean(yMat) # 数据标准化
    # print(yMean)
    yMat = yMat - yMean
    # print(xMat)
    #regularize X's
    xMeans = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat - xMeans) / xVar #（特征-均值）/方差
    numTestPts = 30
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts): # 测试不同的lambda取值，获得系数
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat
xArr = data.iloc[:,:-1]
yArr = data.iloc[:,-1]
# print(xArr,yArr)
ridgeWeights = ridgeTest(xArr,yArr)
# print(ridgeWeights)
plt.plot(ridgeWeights)
plt.show()


# In[ ]:




