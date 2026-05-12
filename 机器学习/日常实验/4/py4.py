#SVM
import numpy as np
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split
import math

# 数据准备
minist_path = r".\data\MNIST.mat"
lung_path = r".\data\lung.mat"
yale_path = r".\data\Yale.mat"
KERNEL = ['linear', 'rbf']


# 加载数据
def create_data(path):
    data = loadmat(path)
    data_x = data["X"]
    data_y = data["Y"][:, 0]
    data_y -= 1
    Data = np.array(data_x)
    Label = np.array(data_y)
    return Data, Label


def laplace(X1, X2):
    K = np.zeros((len(X1), len(X2)), dtype=np.float)
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i][j] = math.exp(-math.sqrt(np.dot(X1[i] - X2[j], (X1[i] - X2[j]).T))/2)
    return K


def classify(path, kernel):
    X, y = create_data(path)
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.333, random_state=233)
    # 训练svm分类器
    classifier = svm.SVC(C=2, kernel=kernel, gamma=10, decision_function_shape='ovr')  
    classifier.fit(train_data, train_label.ravel()) 
    # 计算svc分类器的准确率
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))


if __name__ == '__main__':
    print('yale: ')
    print('linear: ')
    classify(yale_path, KERNEL[0])
    print('rbf: ')
    classify(yale_path, KERNEL[1])
    print('laplace:')
    classify(yale_path, laplace)
    print('-----------------')
    print('lung: ')
    print('linear: ')
    classify(lung_path, KERNEL[0])
    print('rbf: ')
    classify(lung_path, KERNEL[1])
    print('laplace:')
    classify(lung_path, laplace)
    print('-----------------')
    print('minist: ')
    print('linear: ')
    classify(minist_path, KERNEL[0])
    print('rbf: ')
    classify(minist_path, KERNEL[1])
    print('laplace:')
    classify(minist_path, laplace)
