#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


import scipy.io as io
data=io.loadmat("./datasets/data.mat")
# feature,label = data["X"],data["Y"]
print(data)


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# implement stand regress
def standRegress(xArr,yArr):
    # 将数组转换为矩阵
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat # 计算xTx的
    if np.linalg.det(xTx) == 0.0:
        print('xTx不能求逆矩阵')
        return
    theta = xTx.I * (xMat.T * yMat)
    yHat = xMat*theta
    return yHat
 
# deal with data
xArr = []
yArr = []
for d in data["data"]:
    # print(data)
    xTmp = []
    yTmp = []
    xTmp.append(d[0])
    xTmp.append(d[1])
    xTmp.append(d[2])
    xTmp.append(d[3])
    xTmp.append(d[4])
    xTmp.append(d[5])
    xTmp.append(d[6])
    yTmp.append(d[7])
    xArr.append(xTmp)
    yArr.append(yTmp)

# print(xArr[0:2])
print(yArr)
# ws = standRegress(xArr,yArr)
# print(ws)
yHat = standRegress(xArr,yArr)
xMat = np.mat(xArr)
yMat = np.mat(yArr)
# print(yMat.T[0,:].flatten().A[0])
plt.scatter(xMat[:,1].flatten().A[0],yMat.T[0,:].flatten().A[0]) # real data
plt.plot(xMat[:,1],yHat,'r-') # predict data
plt.show()


# In[ ]:




