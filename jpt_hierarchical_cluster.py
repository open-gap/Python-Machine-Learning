#%% 导入头文件
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

#%% 随机生成聚类数据
np.random.seed(123)
variables = ['X', 'Y', "Z"]
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
df

#%% 计算距离矩阵
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), 
    columns=labels, index=labels)
row_dist

#%% 计算关联矩阵(linkage matrix)并转化为DataFrame格式
# 以下计算方式任选其一
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
# row_clusters = linkage(df.values, method='euclidean', method='complete')
row_clusters = pd.DataFrame(row_clusters, 
    columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'], 
    index=['cluster %d' % (i+1) for i in range(row_clusters.shape[0])])
row_clusters

#%% 可视化聚类结果，默认采用自动彩色绘制，删除注释代码可用单色绘制
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['gray'])
row_dendr = dendrogram(row_clusters, labels=labels, 
    # color_threshold=np.inf
    )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

#%% 将树状图和热力图结合起来
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')
df_rowclust = df.ix[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

#%% 使用sklearn进行凝聚聚类
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', 
    linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)