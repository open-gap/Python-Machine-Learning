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

#%% 流水线集成数据处理与分类操作
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([
    ('scl', StandardScaler()), 
    ('pca', PCA(n_components=2)), 
    ('clf', LogisticRegression(random_state=1))
])
pipe_lr.fit(X_train, Y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, Y_test))

#%% 使用k折交叉验证训练流水线模型
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold.split(X_train, Y_train)):
    pipe_lr.fit(X_train[train], Y_train[train])
    score = pipe_lr.score(X_train[test], Y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist: %s, Acc: %.3f' % (k+1, 
        np.bincount(Y_train[train]), score))
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#%% 使用sklearn库的k折交叉验证评分函数直接评分
from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    estimator=pipe_lr, 
    X=X_train, 
    y=Y_train, 
    cv=10, 
    n_jobs=1
)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))