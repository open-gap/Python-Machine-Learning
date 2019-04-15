#%% 导入头文件
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

#%% 获得数据集并划分训练数据和测试数据
df_wine = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-'
    + r'learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 
    'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Noflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 
    'Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
X = df_wine[['Alcohol', 'Hue']].values
Y = df_wine['Class label'].values
Y = LabelEncoder().fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=1
)

#%% 建立分类器模型并对比不同模型的结果及绘制决策区域
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, 
    learning_rate=0.1, random_state=0)

tree = tree.fit(X_train, Y_train)
Y_train_pred = tree.predict(X_train)
Y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(Y_train, Y_train_pred)
tree_test = accuracy_score(Y_test, Y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (
    tree_train, tree_test))

ada = ada.fit(X_train, Y_train)
Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
tree_train = accuracy_score(Y_train, Y_train_pred)
tree_test = accuracy_score(Y_test, Y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (
    tree_train, tree_test))

X_min = X_train[:, 0].min() - 1
X_max = X_train[:, 0].max() + 1
Y_min = X_train[:, 1].min() - 1
Y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1), 
    np.arange(Y_min, Y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', 
    sharey='row', figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, Y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 
        c='blue', marker='^')
    axarr[idx].scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 
        c='red', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
plt.show()