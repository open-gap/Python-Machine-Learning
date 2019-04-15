#%% 导入头文件
import operator
import numpy as np 
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt 
from sklearn.externals import six
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, cross_val_score

#%% 定义多数投票分类器
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = \
            {key:value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, Y):
        self.lablenc_ = LabelEncoder().fit(Y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitter_clf = clone(clf).fit(X, self.lablenc_.transform(Y))
            self.classifiers_.append(fitter_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = \
                np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)), 
                axis=1, arr=predictions
            )
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
        return out

#%% 获取鸢尾花数据集数据并划分训练集与测试集
iris = datasets.load_iris()
X, Y = iris.data[50:, [1, 2]], iris.target[50:]
Y = LabelEncoder().fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=1
)

#%% 利用10折交叉验证验证不同分类器分类效果
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
clf_labels = ['Logistic Regerssion', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=Y_train, cv=10, scoring='roc_auc'
    )
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' 
        % (scores.mean(), scores.std(), label))

#%% 利用多数投票分类器组合模型结果
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=Y_train, cv=10, scoring='roc_auc'
    )
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' 
        % (scores.mean(), scores.std(), label))

#%% 绘制不同分类器的ROC曲线
colors = ['red', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    Y_pred = clf.fit(X_train, Y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, 
        label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#%% 可视化分类器分类结果
X_train_std = StandardScaler().fit_transform(X_train)
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), 
    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', 
    sharey='row', figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, Y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[Y_train == 0, 0], 
        X_train_std[Y_train == 0, 1], c='blue', marker='^', s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[Y_train == 1, 0], 
        X_train_std[Y_train == 1, 1], c='red', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -4.5, s='Sepal width [Standardized]', 
    ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, s='Petal length [Standardized]', 
    ha='center', va='center', fontsize=12, rotation=90)
plt.show()

#%% 使用网格搜索查找模型的最优参数
params = {'decisiontreeclassifier__max_depth': [1, 2], 
    'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, \
    cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, Y_train)
print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %0.3f' % grid.best_score_)