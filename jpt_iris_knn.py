#%% 导入头文件
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn import datasets 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
print('导入头文件成功！')

#%% 导入鸢尾花数据集，进行数据的基本分析
iris_data = datasets.load_iris() #读取鸢尾花数据集
array_X = iris_data.data #读取鸢尾花不同特征量
array_Y = iris_data.target #读取鸢尾花种类代号，分别为0、1、2
list_name = iris_data.target_names #读取鸢尾花种类名称
print('鸢尾花数据集的分类种类为：', list_name)

#%% 由系统导入的类字典数据类型转化为Datafram数据类型
# 创建四个特征的名称用于构建datafram类型变量
attribute = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data_df = pd.DataFrame(array_X, columns=attribute)
print(data_df.info())
print('鸢尾花数据集中缺失数据情况为：\n', data_df.isnull().sum())
# 给datafram数据类型增加变量
data_df['target'] = pd.Series((array_Y))
data_df['class'] = data_df['target'].map(
    {0:'setosa', 1:'versicolor', 2:'virginica'})
data_df.tail()

#%% 绘图对鸢尾花数据集进行分析
sns.pairplot(data_df, vars=['sepal_length', 'sepal_width'],
    hue='class', palette='husl')
sns.pairplot(data_df, vars=['petal_length', 'petal_width'], 
    hue='class', palette='husl')

#%% 根据图形决定选取鸢尾花数据集的petal_length和Petal_width两个特征进行分类
X = data_df[['petal_length', 'petal_width']].values
Y = data_df.target.values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=None
)
print('数据分类完成！')

#%% 执行KNN分类算法对鸢尾花数据集进行分类
def KNNClassify (newdata, dataSet, labels, k):
    if (newdata.size != newdata.shape[0]):
        newdata = newdata[0, :] #每次函数仅处理一行数据
    # 使用欧式距离，即圆形选择框
    distance = np.sum((newdata - dataSet) **2, axis=1)
    # 使用绝对值距离，即菱形选框
    # distance = np.sum(np.abs(newdata - dataset), axis=1)
    sort = distance.argsort() #将距离数组按照从小到大的顺序排列并返回序号
    count = np.zeros(3)
    for i in range(k):
        count[labels[sort[i]]] += 1
    return count.argmax()

predit = np.zeros(Y_test.shape)
for i in range(X_test.shape[0]):
    predit[i] = KNNClassify(X_test[i, :], X_train, Y_train, 5)
print('KNN算法的准确率为：%.2f%%' % (accuracy_score(Y_test, predit) * 100))
sns.heatmap(confusion_matrix(Y_test, predit), annot=True, fmt='d')

#%% 绘制分类结果图
plt.figure(figsize=(8, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predit, marker='o')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('KNN algorithm prediction')