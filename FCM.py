import numpy as np 
import matplotlib.pyplot as plt 

# 使用软聚类的典型算法模糊C-means(FCM)算法分类数据
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
