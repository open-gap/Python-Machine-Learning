#%% 头文件
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%% 特殊函数
#######################################################################################
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), 
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    X_test = X[test_idx, :]
    # y_test = y[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, 
            c=cmap(idx), marker=markers[idx], label=cl)
    
    if test_idx:
        X_test = X[test_idx, :]
        # y_test = y[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, 
            linewidths=1, marker='o', s=55, label='test set')
#######################################################################################

#%% 输入数据处理
row_data, row_target = datasets.load_wine(True)
X_train, X_test, Y_train, Y_test = train_test_split(
    row_data, 
    row_target, 
    test_size=0.3, 
    random_state=0
)
sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#%%
X_test_std = StandardScaler().fit_transform(X_test)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(3):
    mean_vecs.append(np.mean(X_train_std[Y_train == label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label - 1]))

#%%
d = 13
s_w = np.zeros((d, d))
for label, mv in zip(range(3), mean_vecs):
    class_scatter = np.cov(X_train_std[Y_train == label].T)
    s_w += class_scatter
mean_overall = np.mean(X_train_std, axis=0)
s_b = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = np.sum(row_target == i)
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

#%%
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', 
    label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', 
    label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend()
plt.show()

#%%
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, 
    eigen_pairs[1][1][:, np.newaxis].real))
X_train_lda = X_train_std.dot(w)
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(Y_train), colors, markers):
    plt.scatter(X_train_lda[Y_train == l, 0], X_train_lda[Y_train == l, 1], 
        c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, Y_train)
lr = LogisticRegression().fit(X_train_lda, Y_train)
plot_decision_regions(X_train_std, Y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

#%%
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, Y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()