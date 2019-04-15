#%% 导入头文件
import numpy as np 
from matplotlib import cm 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples

#%% 生成分类数据
X, Y = make_blobs(n_samples=150, n_features=2, centers=3, 
    cluster_std=0.5, shuffle=True, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', s=50)
plt.grid()
plt.show()

#%% 使用k-means聚类分类数据集
km = KMeans(n_clusters=3, init='random', n_init=10, 
    max_iter=300, tol=1e-4, random_state=0)
Y_km = km.fit_predict(X)
plt.scatter(X[Y_km == 0, 0], X[Y_km == 0, 1], 
    s=50, c='lightgreen', marker='s', label='cluster 1')
plt.scatter(X[Y_km == 1, 0], X[Y_km == 1, 1], 
    s=50, c='orange', marker='o', label='cluster 2')
plt.scatter(X[Y_km == 2, 0], X[Y_km == 2, 1], 
    s=50, c='lightblue', marker='v', label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
    s=250, c='red', marker='*', label='centroids')
plt.legend()
plt.grid()
plt.show()

#%% 使用软聚类的典型算法模糊C-means(FCM)算法分类数据
class FuzzyCMeans(object):
    def __init__(self, n_clusters, fuzzifier=2, max_iter=300, tol=1e-4):
        assert (fuzzifier >= 1), 'fuzziness coefficient sets too small'
        self.n_clusters = n_clusters
        self.m = fuzzifier
        self.max_iter = max_iter
        self.tol = tol
        self.costs = []
        self.center_list = []

    def __translate_type(self, point_list):
        if isinstance(point_list, np.ndarray):
            point_list_temp = []
            for i in range(point_list.shape[0]):
                point_list_temp.append(point_list[i, :])
            point_list = point_list_temp
        elif not isinstance(point_list, list):
            print('Input point type error!')
            return None
        return point_list

    def __cost(self, point_list, center_list, w_list):
        distance_array = np.apply_along_axis(self.__square_norm, 
            1, point_list, center_list)
        J_m = np.sum(np.array(w_list) * np.array(distance_array))
        return J_m

    def __square_norm(self, point, center_list):
        output_list = []
        for center_point in center_list:
            norm = np.linalg.norm(point - center_point)
            output_list.append(np.power(norm, 2))
        return output_list

    def __calc_w(self, point_list, center_list):
        w_list = np.zeros((len(point_list), self.n_clusters))
        for j in range(len(point_list)):
            for p in range(self.n_clusters):
                distance_p = np.linalg.norm(point_list[j] - center_list[p])
                w_ij = 0.0
                for i in range(self.n_clusters):
                    distance_i = np.linalg.norm(point_list[j] - center_list[i])
                    w_ij += np.power(
                        distance_p / distance_i, 2.0 / (self.m - 1.0))
                if w_ij > 0.0:
                    w_list[j, p] = 1.0 / w_ij
                else:
                    w_list[j, p] = 10000.0
        return w_list

    def __calc_center(self, point_list, w_list):
        center_list = []
        point_array = np.array(point_list)
        for i in range(self.n_clusters):
            w_xj = np.array(w_list)[:, i].reshape(-1, 1)
            center_point = np.sum(
                w_xj * point_array, axis=0) / np.sum(w_xj)
            center_list.append(center_point)
        return center_list

    def __select_center(self, point_list):
        center_list = []
        select_list = []
        num = np.random.randint(len(point_list))
        for _ in range(self.n_clusters):
            center_list.append(point_list[num])
            select_list.append(num)
            distance_list = []
            for i, point in enumerate(point_list):
                if i in select_list:
                    distance_list.append(0.0)
                else:
                    distance = np.linalg.norm(center_list[-1] - point)
                    distance_list.append(np.power(distance, 2))
            prob = distance_list / np.sum(distance_list)
            num = np.random.choice(len(point_list), p=prob)
        return center_list

    def fit(self, point_list):
        point_list = self.__translate_type(point_list)
        if point_list is None:
            return None
        center_list = self.__select_center(point_list)
        w_list = self.__calc_w(point_list, center_list)
        cost = self.__cost(point_list, center_list, w_list)
        costs = [cost]
        for _ in range(self.max_iter):
            center_list = self.__calc_center(point_list, w_list)
            w_list = self.__calc_w(point_list, center_list)
            cost = self.__cost(point_list, center_list, w_list)
            if np.abs(costs[-1] - cost) <= self.tol:
                self.center_list = center_list
                self.costs = costs
                return self
            else:
                costs.append(cost)
        self.costs = costs
        self.center_list = center_list
        return self

    def predict(self, point_list):
        point_list = self.__translate_type(point_list)
        if point_list is None:
            return [None]
        else:
            w_list = self.__calc_w(point_list, self.center_list)
            results = np.argmax(w_list, axis=1)
            return results

    def fit_predict(self, poinit_list):
        if self.fit(poinit_list) is None:
            return [None]
        else:
            return self.predict(poinit_list)

    def fit_plot(self, point_list, plot_cost=False):
        if self.fit(point_list) is None:
            return None
        results = self.predict(point_list)
        colors = ['red', 'yellow', 'blue', 'green', 'lightgray', 'c', 
            'pink', 'orange', 'greenyellow', 'm']
        markers = ['D', 'o', 'v', '^', '<', '>', 's', 'p', '*', '+']
        if not plot_cost:
            plt.figure(figsize=(6, 5))
        elif plot_cost:
            plt.figure(figsize=(11, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(len(self.costs)) + 1, self.costs)
            plt.xlabel('Iter')
            plt.ylabel('Cost')
            plt.title('Cost function curve')
            plt.subplot(1, 2, 2)
        point_array = np.array(point_list)
        for i in np.unique(results):
            plt.scatter(point_array[results == i, 0], 
                point_array[results == i, 1],  s=50, c=colors[i], 
                marker=markers[i], label='cluster %d' % i)
        plt.title('Cluster result')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        return self

#%% 使用FCM算法分类数据
point_list = []
for i in range(X.shape[0]):
    point_list.append(X[i, :])
fcm = FuzzyCMeans(n_clusters=3)
fcm.fit_predict(point_list)

#%% 使用聚类结果的簇内误差平方和作为评价聚类结果的标准
# 通过评价标准判断k-means++算法的最适合的分类数目n_clusters
print('Distortion: %.3f' % km.inertia_)
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, 
        random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#%% 通过轮廓图定量分析聚类质量
km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4, 
    random_state=0)
Y_km = km.fit_predict(X)
cluster_labels = np.unique(Y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, Y_km, metric='euclidean')
Y_ax_lower, Y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[Y_km == c]
    c_silhouette_vals.sort()
    Y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(Y_ax_lower, Y_ax_upper), c_silhouette_vals, height=1.0, 
        edgecolor='none', color=color)
    yticks.append((Y_ax_lower + Y_ax_upper) / 2)
    Y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals) # =silhouette_scores(X, Y_km, ...)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

#%% 采用两个中心点聚类算法比较聚类效果较差情况下的轮廓图
km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-4, 
    random_state=0)
Y_km = km.fit_predict(X)
plt.scatter(X[Y_km == 0, 0], X[Y_km == 0, 1], s=50, c='lightgreen', marker='s', 
    label='Cluster 1')
plt.scatter(X[Y_km == 1, 0], X[Y_km == 1, 1], s=50, c='orange', marker='o', 
    label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, 
    c='red', marker='*', label='Centroids')
plt.legend()
plt.grid()
plt.show()

cluster_labels = np.unique(Y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, Y_km, metric='euclidean')
Y_ax_lower, Y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[Y_km == c]
    c_silhouette_vals.sort()
    Y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(Y_ax_lower, Y_ax_upper), c_silhouette_vals, height=1.0, 
        edgecolor='none', color=color)
    yticks.append((Y_ax_lower + Y_ax_upper) / 2)
    Y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals) # =silhouette_scores(X, Y_km, ...)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()