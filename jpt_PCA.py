#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine 
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 

#%%
# row_wine = load_wine()
# target = row_wine['target'].reshape(row_wine['target'].size, 1)
# df_wine = np.hstack((target, row_wine['data']))
# df_wine = pd.DataFrame(df_wine)
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/' + 
    'machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 
                    'Malic acid', 'Ash', 
                    'Alcalinity of ash', 'Magnesium', 
                    'Total phenols', 'Flavanoids', 
                    'Nonflavanoid phenols', 
                    'Proanthocyanins', 
                    'Color intensity', 'Hue', 
                    'OD280/OD315 of diluted wines', 
                    'Proline']
df_wine.tail()

#%%
X, Y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0
)
stdsc = StandardScaler().fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', 
    label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', 
    label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

#%%
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
    for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
    eigen_pairs[1][1][:, np.newaxis]))
print(w)

#%%
X_train_pca = X_train_std.dot(w) #利用原始数据卷积特征向量矩阵降维
colors = ['red', 'lightgreen', 'blue']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(Y_train), colors, markers):
    plt.scatter(X_train_pca[Y_train == l, 0], 
        X_train_pca[Y_train == l, 1], 
        c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

#%% 利用sklearn库自带的PCA算法计算
def plot_decision_regions(X, Y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'lightgreen', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
        np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1], 
            alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression().fit(X_train_pca, Y_train)
plot_decision_regions(X_train_pca, Y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()

#%% 绘制测试集情况
plot_decision_regions(X_test_pca, Y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()

# 备注：PCA中n_components设置为None则按照特征方差贡献率递减顺序返回所有主成分