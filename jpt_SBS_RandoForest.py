#%%
import numpy as np 
import pandas as pd 
from sklearn.base import clone 
import matplotlib.pyplot as plt 
from itertools import combinations 
from sklearn.datasets import load_wine 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#%%
row_wine = load_wine()
target = row_wine['target'].reshape(row_wine['target'].size, 1)
df_wine = np.hstack((target, row_wine['data']))
df_wine = pd.DataFrame(df_wine)
# df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/' + 
#     'machine-learning-databases/wine/wine.data', header=None)
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

#%% 通过序列特征选择算法排除不重要特征
class SBS():
    def __init__(self, estimator, k_features, 
        scoring=accuracy_score, test_size=0.25, random_state=0):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, 
            random_state=self.random_state
        )

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, Y_train, X_test, 
            Y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, Y_train, 
                    X_test, Y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, Y_train, X_test, Y_test, indices):
        self.estimator.fit(X_train[:, indices], Y_train)
        Y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(Y_test, Y_pred)
        return score

#%%
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, Y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

#%%
k9 = list(sbs.subsets_[4])
print(df_wine.columns[1:][k9])

#%%
knn.fit(X_train_std, Y_train)
print('Training accuracy:', knn.score(X_train_std, Y_train))
print('Test accuracy:', knn.score(X_test_std, Y_test))

#%%
knn.fit(X_train_std[:, k9], Y_train)
print('Training accuracy:', knn.score(X_train_std[:, k9], Y_train))
print('Test accuracy:', knn.score(X_test_std[:, k9], Y_test))

#%% 通过随机森林判定特征的重要性，注意输入数据不需要标准化
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, Y_train)
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]
for f in range(X_train.shape[1]):
    print('%2d) %-*s %f' % (f+1, 30, feat_labels[f], importance[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importance[indices], 
    color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
