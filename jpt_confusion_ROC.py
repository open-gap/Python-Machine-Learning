#%% 通用头文件
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

#%% 加载威斯康辛乳腺癌数据集并区分数据集与训练集
df = pd.read_csv(
    r'https://archive.ics.uci.edu/ml/machine-learning' 
    + r'-databases/breast-cancer-wisconsin/wdbc.data', 
    header= None)
X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values
Y = LabelEncoder().fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1
)

#%% 构建SVM分类器流水线模型并计算混淆矩阵
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
pipe_svc = Pipeline([
    ('scl', StandardScaler()), 
    ('clf', SVC(random_state=1))
])
pipe_svc.fit(X_train, Y_train)
Y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print('Confusion matrix:')
print(confmat)

#%% 绘制混淆矩阵图形
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3) #绘制矩阵图像
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(j, i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

#%% 计算分类器结果的准确率、召回率和F1分数
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=Y_pred))
print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=Y_pred))
print('F1: %.3f' % f1_score(y_true=Y_test, y_pred=Y_pred))

#%% 网格搜索中scoring搜索标准还可以指定不同值或通过make_scorer构造参数，例如
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score
# scorer = make_scorer(f1_score, pos_label=0)
# gs = GridSearchCV(
#     estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10
# )

#%% 绘制受试者工作特征曲线(ROC)
from scipy import interp #一维线性插值
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
pipe_lr = Pipeline([
    ('scl', StandardScaler()), 
    ('clf', LogisticRegression())
])
X_train2 = X_train[:, [4, 14]] #只使用数据集中的两个特征估计分类结果
cv = StratifiedKFold(n_splits=3, random_state=1) #只使用了三个分块
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv.split(X_train2, Y_train)):
    probas = pipe_lr.fit(X_train2[train], Y_train[train])
    probas = probas.predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(Y_train[test], probas[:, 1], pos_label=1)

    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, 
        label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', 
    label='Random Guessing')
mean_tpr /= (i + 1)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'r--', 
    label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='green', 
    label='Perfect Performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc='lower right')
plt.show()

#%% 利用sklearn库直接计算ROC AUC得分
from sklearn.metrics import roc_auc_score, accuracy_score
pipe_svc = pipe_svc.fit(X_train2, Y_train)
Y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])
print('ROC AUC: %.3f' % roc_auc_score(Y_test, Y_pred2))
print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred2))

#%% 绘制准确率-召回率曲线(PRC)
from sklearn.metrics import precision_recall_curve
fig = plt.figure(figsize=(8, 6))

for i, (train, test) in enumerate(cv.split(X_train2, Y_train)):
    probas = pipe_lr.fit(X_train2[train], Y_train[train])
    probas = probas.predict_proba(X_train2[test])
    pre, rec, thresholds = precision_recall_curve(
            Y_train[test], probas[:, 1], pos_label=1
            )
    plt.plot(rec, pre, lw=1, 
        label='PRC fold %d' % (i + 1))

plt.plot([0, 1], [0, 1], 'g--', label='Max F1 Score Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Precision Rate')
plt.ylabel('Recall Rate')
plt.title('Precision Recall Curve')
plt.legend(loc='lower right')
plt.show()