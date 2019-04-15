import time
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from sklearn.tree import export_graphviz
# from sklearn.tree import DecisionTreeClassifier

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

# np.random.seed(0)
# X_xor = np.random.randn(200, 2)
# Y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# Y_xor = np.where(Y_xor, 1, -1)
# plt.scatter(X_xor[Y_xor == 1, 0], X_xor[Y_xor == 1, 1], c='b', marker='x', label='1')
# plt.scatter(X_xor[Y_xor == -1, 0], X_xor[Y_xor == -1, 1], c='r', marker='s', label='-1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()

row_data, row_target = datasets.load_iris(True)
X_train, X_test, Y_train, Y_test = train_test_split(
    row_data[:, [0, 2]], 
    row_target, 
    test_size=0.33, 
    random_state=0
)

# plt.scatter(X_train[:, 0], X_train[:, 1])
# plt.show()

# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, 
#   random_state=0)
# tree.fit(X_train, Y_train)

# forest = RandomForestClassifier(
#     criterion='entropy', 
#     n_estimators=10, 
#     random_state=1, 
#     n_jobs=4
# )
# forest.fit(X_train, Y_train)

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, Y_train)

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))
plot_decision_regions(X_combined, Y_combined, classifier=knn, 
    test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
