import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from numpy.random import random
from sklearn.model_selection import train_test_split

##################################定##义##函##数##区##########################################

#sigmoid函数，神经网络非线性环节
def sigmoid (x_input, theta):
    return 1 / (1 + np.exp(- x_input.dot(theta)))

#三层前馈网络的计算，获得前向预测值
def feedforward_three (x_input, theta_one, theta_two):
    middle = sigmoid(x_input, theta_one.T)
    middle_after = np.column_stack((np.ones(middle.shape[0]), middle))
    return middle, sigmoid(middle_after, theta_two.T)

#代价函数，用于计算预测值与实际值的误差
def cost (h_theta, y, theta_one, theta_two, lamb):
    m = y.shape[0]
    if y.size == y.shape[0]: #对于模型输出为一维时的j_raw处理方法
        j_raw = np.sum(np.diag(- np.log(h_theta) * y - (1 - y) * np.log(1 - h_theta))) /m
    else: #对于模型输出为二维时的j_raw处理方法
        j_raw = np.sum(- np.log(h_theta) * y - (1 - y) * np.log(1 - h_theta)) /m
    regular_theta = (np.sum(theta_one * theta_one) + np.sum(theta_two * theta_two)) *lamb /(2 *m)
    return j_raw + regular_theta

#三层神经网络梯度计算函数，用于计算参数更新时的负梯度值
def gradient (x, middle, y, h_theta, theta_one, theta_two):
    if y.size == y.shape[0]: #对于模型输出为一维向量时
        delta_three = (h_theta.T - y).T #矩阵和向量的加减法的特殊（奇葩）处理方式
    else: #对于模型输出为二维或多维矩阵时
        delta_three = h_theta - y
    delta_two = delta_three.dot(theta_two)[:, 1:] * middle * (1 - middle)
    middle_after = np.column_stack((np.ones(middle.shape[0]), middle))
    gradient_two = delta_three.T.dot(middle_after) / x.shape[0]
    gradient_one = delta_two.T.dot(x) / x.shape[0]
    return gradient_one, gradient_two

#神经网络主函数，计算神经网络层间参数
def neural_network (x_input, y_input, middle_layer, output_layer, max_iter, error, lamb, mu):
    theta_one = random(size = (middle_layer, x_input.shape[1] + 1))
    theta_two = random(size = (output_layer, middle_layer + 1))
    X = np.column_stack((np.ones(x_input.shape[0]), x_input))
    middle, h_theta = feedforward_three(X, theta_one, theta_two)
    error_list = [cost(h_theta, y_input, theta_one, theta_two, mu)]
    print('第一次计算梯度前的Cost Function值为：', error_list[0])
    while (error_list[-1] > error and max_iter > 0):
        gradient_one, gradient_two = gradient(X, middle, y_input, h_theta, theta_one, theta_two)
        theta_one -= gradient_one + lamb * theta_one / X.shape[0]
        theta_two -= gradient_two + lamb * theta_two / X.shape[0]
        middle, h_theta = feedforward_three(X, theta_one, theta_two)
        error_list.append(cost(h_theta, y_input, theta_one, theta_two, mu))
        max_iter -= 1
    return theta_one, theta_two, error_list

#测试三层神经网络训练结果准确度
def accuracy (x_input, y_input, theta_one, theta_two):
    X = np.column_stack((np.ones(x_input.shape[0]), x_input))
    middle, h_theta = feedforward_three(X, theta_one, theta_two)
    y_predict = np.zeros(y_input.shape)
    if y_input.size == y_input.shape[0]: #对于模型输出为一维情况时
        y_predict[h_theta[:, 0] >= 0.5] = 1
    else: #对于模型输出为二维情况时的处理方法
        y_predict[h_theta >= 0.5] = 1
    return np.sum(y_predict == y_input) / y_predict.size

########################################主##函##数##区##############################################

#原始数据读取与数据预处理，排除字符串数据和缺失数据
datafram = pd.read_csv(r'D:/project/machine_learning_databases/indian_liver_patient.csv')
datafram['Gender'] = datafram['Gender'].replace(['Male', 'Female'], [1, 0])
datafram.drop(datafram.index[datafram['Albumin_and_Globulin_Ratio'].isnull()], inplace = True)

#由读取数据获取所需的训练数据集和测试数据集
X_raw = np.array(datafram.iloc[:, :-1])
Y_raw = np.array(datafram.iloc[:, -1])
X_raw = (X_raw - X_raw.mean(axis = 0)) / X_raw.std(axis = 0)

#######################################################
output_layer = 1 #设定输出层维数，这将影响后续代码的执行
#######################################################
if output_layer == 1:
    y_out = Y_raw - 1 #对于原始的一维输出转化为矩阵运算
else:
    #针对仅有一维输出的数据提高到二维或多维输出
    y_out = np.zeros((Y_raw.shape[0], 2))
    k = 0
    for i in Y_raw:
        y_out[k, i - 1] = 1
        k += 1
#分割原始数据
X_train, X_test, Y_train, Y_test = train_test_split(
    X_raw, y_out, test_size = 0.33, random_state = None
)

#根据获得的训练数据训练获取参数θ
hidden_layer = 8
max_iter = 200
max_error = 1e-4
learning_rate = 0.3
regular_rate = 0.3
theta_one, theta_two, error_list = neural_network(
    X_train, Y_train, hidden_layer, output_layer, 
    max_iter, max_error, learning_rate, regular_rate
)

#评估神经网络训练结果并输出
iter_num = len(error_list) - 1
print('三层神经网络的两个参数Θ维度分别为：', theta_one.shape, ';', theta_two.shape)
print('迭代次数为：', iter_num, '；最后一次的Cost Function值为：', error_list[-1])
print('模型输出结果= {0:d} 维时的'.format(output_layer), end = '')
print('训练结果准确率为：{0:.4f}%'.format(accuracy(X_test, Y_test, theta_one, theta_two) * 100))
# print('Cost Function值为：', error_list[1:])
zhfont = mpl.font_manager.FontProperties(fname = r'C:\Windows/Fonts/msyh.ttc')
plt.plot(np.arange(iter_num) + 1, error_list[1:])
plt.title('三层神经网络训练过程误差下降示意图', fontproperties = zhfont)
plt.xlabel('迭代次数', fontproperties = zhfont)
plt.ylabel('Cost Function值', fontproperties = zhfont)
plt.grid(True)
plt.show()