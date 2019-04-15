import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron # sklearn中的logistic回归
from sklearn.preprocessing import StandardScaler # sklearn中参数初始化
from sklearn.linear_model import LinearRegression # sklearn中的线性回归
from sklearn.preprocessing import PolynomialFeatures # sklearn中的多项式变量生成
from sklearn.model_selection import train_test_split # sklearn中的随机数据分组
from sklearn.linear_model.logistic import LogisticRegression # sklearn中的logistic回归

# 读取数据并划分为输入向量X_raw和输出向量Y_raw
datafram = pd.read_csv(r'D:\project/machine_learning_databases/indian_liver_patient.csv')
datafram.head() #输出读取数据前5行，用于验证数据读取是否正确
datafram['Gender'] = datafram['Gender'].replace(['Male', 'Female'], [1, 0])
datafram.drop(datafram.index[datafram['Albumin_and_Globulin_Ratio'].isnull()], inplace = True)
X_raw = np.array(datafram.iloc[:, :-1])
Y_raw = np.array(datafram.iloc[:, -1])
# 将输入向量和输出向量标准化，利用sklearn.preprocessing的StandardScalar方法
X_scalar = StandardScaler().fit(X_raw)
X_standard = X_scalar.transform(X_raw)
Y_standard = Y_raw - 1
# 将所以数据随机分类成训练数据集和测试数据集两类
x_train, x_test, y_train, y_test = train_test_split(
    X_standard, Y_standard, test_size = 0.33, random_state = None
)

# 利用sklearn的（多元）线性回归预测结果
linear_regression = LinearRegression().fit(x_train, y_train)
linear_predict = linear_regression.predict(x_test)
y_pred = np.zeros(linear_predict.shape)
y_pred[linear_predict >= 0.5] = 1
print('sklearn多元线性回归的预测准确度为：{0:.4f}%'.format(np.sum(y_pred == y_test) /y_test.size *100))
# 绘制测试预测值与实际值的差值图
# plt.scatter(linear_predict, linear_predict - y_test, c='red', marker='o')
# plt.hlines(y=0, xmin=min(linear_predict), xmax=max(linear_predict), lw=2, colors='blue')
# plt.tight_layout()
# plt.show()
# 利用skleran的多项式回归库有单一维度变量生成多维度变量
# polynf = PolynomialFeatures(degree = 3) #设定多项式拟合中多项式的最高次数
# X_quad = polynf.fit_transform(X) #对于X列向量，对其按列生成并返回[1, X, X^2, ..., X^degree]

# 利用sklearn中的logistic回归预测结果
logistic_regression = LogisticRegression().fit(x_train, y_train)
logistic_predict = logistic_regression.predict(x_test)
y_pred = np.zeros(logistic_predict.shape)
y_pred[logistic_predict >= 0.5] = 1
print("sklearn的logistic回归预测准确率为：{0:.4f}%".format(np.sum(y_pred == y_test) /y_test.size *100))

# 利用sklearn的感知器（两层神经网络）预测结果
ppn = Perceptron(max_iter=300, eta0=0.3, random_state=0).fit(x_train, y_train)
ppn_predict = ppn.predict(x_test)
y_pred = np.zeros(ppn_predict.shape)
y_pred[ppn_predict >= 0.5] = 1
print('sklearn的感知器预测准确率为：{0:.4f}%'.format(np.sum(y_test == y_pred) /y_test.size *100))