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

#%% 利用网格搜索查找模型最优参数
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
pipe_svc = Pipeline([
    ('scl', StandardScaler()), 
    ('clf', SVC(random_state=1))
])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {'clf__C':param_range, 'clf__kernel':['linear']}, 
    {'clf__C':param_range, 'clf__gamma':param_range, 'clf__kernel':['rbf']}
]
gs = GridSearchCV(
    estimator=pipe_svc, param_grid=param_grid, 
    scoring='accuracy', cv=10, n_jobs=-1
)
gs = gs.fit(X_train, Y_train)
print('Best Score:', gs.best_score_)
print('Best Params:', gs.best_params_)
clf = gs.best_estimator_
clf.fit(X_train, Y_train)
print('Test accuracy: %.3f' % clf.score(X_test, Y_test))

#%% 利用随机搜索查找模型最优参数
from sklearn.model_selection import RandomizedSearchCV
param_random = {
    'clf__C':param_range, 'clf__gamma':param_range, 
    'clf__kernel':['rbf', 'linear']
}
rs = RandomizedSearchCV(
    estimator=pipe_svc, param_distributions=param_random, 
    scoring='accuracy', cv=10, n_jobs=-1
)
rs.fit(X_train, Y_train)
print('Best Score:', rs.best_score_)
print('Best Params:', rs.best_params_)
clf = rs.best_estimator_
clf.fit(X_train, Y_train)
print('Test accuracy: %.3f' % clf.score(X_test, Y_test))

#%% 使用5x2嵌套交叉验证选择最优的模型
from sklearn.model_selection import cross_val_score
gs = GridSearchCV(
    estimator=pipe_svc, param_grid=param_grid, 
    scoring='accuracy', cv=10, n_jobs=-1
)
scores = cross_val_score(gs, X, Y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#%% 在5x2交叉验证的前提下，使用简单的决策树模型比较SVM模型分类器效果
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0), 
    param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
    scoring='accuracy', cv=5
)
scores = cross_val_score(gs, X, Y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))