import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

def sigmoid (x_input, theta):
    return 1/(1 + np.exp(- x_input.dot(theta)))

def cost (x_input, y_input, theta, lamb):
    m = x_input.shape[0]
    X = np.column_stack((np.ones(m), x_input))
    h_theta = sigmoid(X, theta)
    alpha = np.ones(theta.size)
    alpha[0] = 0
    return ((- y_input.T.dot(np.log(h_theta)) - (1 - y_input).T.dot(np.log(1 - h_theta)) 
    + 0.5 * lamb * theta.T.dot(alpha * theta)) /m)

def gradient (x_input, y_input, theta, lamb):
    m = x_input.shape[0]
    X = np.column_stack((np.ones(m), x_input))
    h_theta = sigmoid(X, theta)
    alpha = np.ones(theta.size)
    alpha[0] = 0
    return (X.T.dot(h_theta - y_input) + lamb * alpha * theta) /m

def fit (x_input, y_input, max_iter, lamb, regular_num):
    theta = np.ones(x_input.shape[1] + 1)
    cost_error = []
    for i in range(max_iter):
        theta -= gradient(x_input, y_input, theta, regular_num)
        cost_error.append(cost(x_input, y_input, theta, regular_num))
    return theta, cost_error

def accuracy (x_input, y_input, theta):
    X = np.column_stack((np.ones(x_input.shape[0]), x_input))
    y = X.dot(theta)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return np.sum(y == y_input) / y.size

datafram = pd.read_csv(r'D:/project/machine_learning_databases/indian_liver_patient.csv')
datafram['Gender'] = datafram['Gender'].replace(['Male', 'Female'], [1, 0])
datafram.drop(datafram.index[datafram['Albumin_and_Globulin_Ratio'].isnull()], inplace = True)
X_raw = np.array(datafram.iloc[:, 0:10])
Y_raw = np.array(datafram.iloc[:, 10])
X_raw = (X_raw - X_raw.mean(axis = 0)) / X_raw.std(axis = 0)
Y_raw = Y_raw - 1
X_train, X_test, Y_train, Y_test = train_test_split(
    X_raw, Y_raw, test_size = 0.33, random_state = None
)

iter_num = 80
learning_rate = 0.3
regular_num = 0.3
theta, cost_error = fit(X_train, Y_train, iter_num, learning_rate, regular_num)
print('训练的结果θ为：', theta)
print('训练的准确度为：{0:.4f}%'.format(accuracy(X_test, Y_test, theta) * 100))

plt.plot(np.arange(iter_num) + 1, cost_error)
zhfont = mpl.font_manager.FontProperties(fname = r'C:/Windows/Fonts/msyh.ttc')
plt.xlabel('迭代次数', fontproperties = zhfont)
plt.ylabel('Cost Funcation值', fontproperties = zhfont)
plt.grid(True)
plt.show()