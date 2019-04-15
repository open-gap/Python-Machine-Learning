#%% 导入头文件
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering

#%% 产生聚类用的数据
X, Y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

#%% 分别使用k-means和基于全连接的层次聚类算法聚类数据
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
km = KMeans(n_clusters=2, random_state=0)
Y_km = km.fit_predict(X)
ax1.scatter(X[Y_km == 0, 0], X[Y_km == 0, 1], c='lightblue', marker='o', 
    s=40, label='Cluster 1')
ax1.scatter(X[Y_km == 1, 0], X[Y_km == 1, 1], c='red', marker='s', s=40, 
    label='Cluster 2')
ax1.set_title('K-means clustering')
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', 
    linkage='complete')
Y_ac = ac.fit_predict(X)
ax2.scatter(X[Y_ac == 0, 0], X[Y_ac == 0, 1], c='lightblue', marker='o', 
    s=40, label='Cluster 1')
ax2.scatter(X[Y_ac == 1, 0], X[Y_ac == 1, 1], c='red', marker='s', s=40, 
    label='CLuster 2')
ax2.set_title('Agglomerative clustering')
plt.legend()
plt.show()

#%% 使用DBSCAN算法聚类数据
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
Y_db = db.fit_predict(X)
plt.scatter(X[Y_db == 0, 0], X[Y_db == 0, 1], c='lightblue', marker='o', 
    s=40, label='Cluster 1')
plt.scatter(X[Y_db == 1, 0], X[Y_db == 1, 1], c='red', marker='s', s=40, 
    label='Cluster 2')
plt.legend()
plt.show()