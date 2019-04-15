#%% 导入头文件
import numpy as np 
from scipy import exp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

#%% 定义RBF核的PCA函数
def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation.

    Parameters
    -------------
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Truning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    -------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_featrures]
        Projected dataset
    '''
    sq_dists = pdist(X, 'sqeuclidean')

    mat_sq_dists = squareform(sq_dists)

    K = exp(- gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eigvals, eigvecs = eigh(K)

    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc

###########################################################################
#%% 示例一：分离半月形数据
from sklearn.datasets import make_moons
X, Y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
plt.show()

#%% 使用标准PCA分离数据
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[Y == 0, 0], X_spca[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[Y == 1, 0], X_spca[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
ax[1].scatter(X_spca[Y == 0, 0], np.zeros((50, 1)) + 0.02, 
    color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[Y == 1, 0], np.zeros((50, 1)) - 0.02, 
    color='green', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
plt.show()

#%% 使用RBF核PCA算法分离数据
from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 2))
ax[0].scatter(X_kpca[Y == 0, 0], X_kpca[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[Y == 1, 0], X_kpca[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[Y == 0, 0], np.zeros((50, 1)) + 0.02, 
    color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[Y == 1, 0], np.zeros((50, 1)) - 0.02, 
    color='green', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_xlabel('PC2')
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

##########################################################################
#%% 分离同心圆
from sklearn.datasets import make_circles
X, Y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
plt.show()

#%% 使用标准PCA方式分离数据
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[Y == 0, 0], X_spca[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[Y == 1, 0], X_spca[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
ax[1].scatter(X_spca[Y == 0, 0], np.zeros((500, 1)) + 0.02, 
    color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[Y == 1, 0], np.zeros((500, 1)) - 0.02, 
    color='green', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
plt.show()

#%% 使用RBF核PCA函数分离数据
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 2))
ax[0].scatter(X_kpca[Y == 0, 0], X_kpca[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[Y == 1, 0], X_kpca[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[Y == 0, 0], np.zeros((500, 1)) + 0.02, 
    color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[Y == 1, 0], np.zeros((500, 1)) - 0.02, 
    color='green', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_xlabel('PC2')
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

#%% 修改RBF核PCA函数使其返回核矩阵的特征值
def Rbf_Kernel_Pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation.

    Parameters
    -------------
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Truning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    -------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_featrures]
        Projected dataset

    lambdas: list
        Eigenvalues

    '''
    sq_dists = pdist(X, 'sqeuclidean')

    mat_sq_dists = squareform(sq_dists)

    K = exp(- gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eigvals, eigvecs = eigh(K)

    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    return alphas, lambdas

#%% 创建新的半月数据并使用新的RBF核PCA函数映射到一维空间中
X, Y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = Rbf_Kernel_Pca(X, gamma=15, n_components=1)
X_new = X[25]   #假定新的数据点
X_proj = alphas[25]  #原始数据点的映射
print('X_new:', X_new, '    X_proj:', X_proj)

def project_X(X_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((X_new - row)**2) for row in X])
    K = np.exp(- gamma * pair_dist)
    return K.dot(alphas / lambdas)

X_reproj = project_X(X_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
print('X_reproj:', X_reproj)
plt.scatter(alphas[Y == 0, 0], np.zeros((50)), 
    color='red', marker='^', alpha=0.5)
plt.scatter(alphas[Y == 1, 0], np.zeros((50)), 
    color='green', marker='o', alpha=0.5)
plt.scatter(X_proj, 0, color='white', 
    label='origainal projection of point X[25]', marker='^', s=300)
plt.scatter(X_reproj, 0, color='blue', 
    label='remapped point X[25]', marker='X', s=100)
plt.legend(scatterpoints=1)
plt.show()

#%% 使用scikit-learn库的核PCA函数分类
from sklearn.decomposition import KernelPCA
X, Y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[Y == 0, 0], X_skernpca[Y == 0, 1], 
    color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[Y == 1, 0], X_skernpca[Y == 1, 1], 
    color='green', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()