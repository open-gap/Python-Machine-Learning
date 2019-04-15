import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from sklearn.model_selection import train_test_split 

def cost (X_raw, Y, sita, Lambda):
    X = np.column_stack((np.ones(np.size(X_raw, 0)), X_raw))
    alpha = np.ones(sita.shape)
    alpha[0] = 0
    return np.sum((X.dot(sita) - Y)**2) / (2 * np.size(X, 0))

def gradient_decent (X, Y, sita, Lambda):
    X = np.column_stack((np.ones(np.size(X, 0)), X))
    alpha = np.ones(sita.shape)
    alpha[0] = 0
    return X.T.dot(X.dot(sita) - Y) / np.size(X, 0)

def fit (X, Y, max_iter, alpha):
    sita = np.ones(np.size(X, 1) + 1)
    cost_value = []
    for i in range(max_iter):
        sita -= alpha * gradient_decent(X, Y, sita, 0.1)
        cost_value.append(cost(X, Y, sita, 0.1))
    plt.plot(np.arange(max_iter) + 1, cost_value)
    zhfont1 = mpl.font_manager.FontProperties(fname = r'C:/Windows/Fonts/msyh.ttc')
    plt.xlabel('迭代次数', fontproperties = zhfont1)
    plt.ylabel('代价函数', fontproperties = zhfont1)
    plt.grid(True)
    plt.show()
    return sita

def accuracy (X, Y, sita):
    X = np.column_stack((np.ones(np.size(X, 0)), X))
    y = X.dot(sita)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return np.sum(Y == y) / y.size

raw_data = pd.read_csv(r'D:/project/machine_learning_databases/indian_liver_patient.csv')
raw_data['Gender'] = raw_data['Gender'].replace(['Male', 'Female'],[1, 0])
raw_data.drop(raw_data.index[raw_data['Albumin_and_Globulin_Ratio'].isnull()], inplace = True)
X_raw = np.array(raw_data.iloc[:, 0:10])
X_raw = (X_raw - X_raw.mean(axis = 0)) / X_raw.std(axis = 0)
Y_raw = np.array(raw_data.iloc[:, 10]) - 1

X_train, X_test, Y_train, Y_test = train_test_split(
    X_raw, Y_raw, test_size = 0.33, random_state = None
)

sita = fit(X_train, Y_train, 30, 0.1)
print('θ参数为：', sita)
print('模型准确度为：{0:.4f}%'.format(accuracy(X_test, Y_test, sita) * 100))