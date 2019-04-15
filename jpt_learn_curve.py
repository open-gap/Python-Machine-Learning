#%% 通用头文件
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

#%% 加载威斯康辛乳腺癌数据集并区分数据集与训练集
df = pd.read_csv(
    r'https://archive.ics.uci.edu/ml/machine-learning' 
    + r'-databases/breast-cancer-wisconsin/wdbc.data', 
    header= None)
X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values
Y = LabelEncoder().fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1
)

#%% 利用学习曲线函数评估模型
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('clf', LogisticRegression(random_state=0))
])
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr, X=X_train, y=Y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='red', marker='o', 
    markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, 
    train_mean - train_std, alpha=0.5, color='red')
plt.plot(train_sizes, test_mean, color='green',linestyle='--',
    marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, 
    test_mean - test_std, alpha=0.5, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#%% 利用验证曲线测试不同参数对模型准确率的影响
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
    estimator=pipe_lr, X=X_train, y=Y_train, param_name='clf__C', 
    param_range=param_range, cv=10
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='red', marker='o', 
    markersize=5, label='Training Accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, 
    alpha=0.3, color='red')
plt.plot(param_range, test_mean, color='green',linestyle='--', 
    marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, 
    alpha=0.3, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.show()