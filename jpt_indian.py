#%% 导入头文件
import numpy as np 
import pandas as pd 
import seaborn as sns #增强型的绘图工具
import matplotlib as mpl 
import matplotlib.pyplot as plt 
print('导入头文件成功！')

#%% CSV文件数据读取
liver_df = pd.read_csv(
    r'D:\project/machine_learning_databases/indian_liver_patient.csv')
print('读取数据成功！')

#%% 数据的显示
# liver_df.head(10)
liver_df.tail(10)

#%% 输出读取到数据的基本情况
print(liver_df.info()) #输出读取数据的基本情况
print(liver_df.columns) #输出数据集的列名称
print(liver_df.isnull().sum()) #输出缺失数据情况

#%% 异常数据采用删除整行数据处理
liver_df.drop(
    liver_df.index[liver_df['Albumin_and_Globulin_Ratio'].isnull()], 
    inplace=True)
print('删除异常数据成功！')

#%% 统计并绘制分组数据情况
LD, NLD = liver_df['Dataset'].value_counts() #统计正常和患病者总人数情况
print('患肝病的总人数：%d人' % LD)
print('未患肝病的总人数：%d人' % NLD)
plt.figure() #统计绘制患病与未患病总人数
sns.countplot(x='Dataset', data=liver_df, label='Count') #利用seaborn库绘制统计图
M, F = liver_df['Gender'].value_counts()
print('男性总人数：%d人' % M)
print('女性总人数：%d人' % F)
plt.figure() #统计绘制性别分布情况
sns.countplot(data=liver_df, x='Gender', label='Count')
plt.figure() #统计绘制患病情况与年龄和性别的关系
sns.pointplot(x='Age', y='Gender', hue='Dataset', data=liver_df) #绘制点图查看趋势

#%% 对数据按照年龄和性别进行分组统计
liver_df[['Gender', 'Age', 'Dataset']].groupby(
    ['Dataset', 'Gender'], as_index=False
).mean().sort_values(by='Gender', ascending=False)

#%%------------------- 绘制年龄、性别与患病与否的关系统计图 --------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.hist, 'Age')
plt.subplots_adjust(top=0.9)
graph.fig.suptitle('Disease by Gender and Age')

#%%------------------- 绘制不同胆红素、年龄、性别和患病的关系 -------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.scatter, 'Direct_Bilirubin', 'Total_Bilirubin', edgecolor='w')
# 由图片可知Direct_Bilirubin和Total_Bilirubin有相关性，需要移除其中一个变量

#%%-------------------- 绘制胆红素两个变量的相关性示意图 ------------------------#
plt.figure()
sns.jointplot('Direct_Bilirubin', 'Total_Bilirubin', data=liver_df, kind='reg')
# 通过将两个变量绘制在相关性图中进一步证明了两个变量具有良好的相关性

#%%-------------------- 讨论两种氨基转氨酶的相关性情况 ---------------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, row='Dataset', col='Gender', margin_titles=True)
graph.map(plt.scatter, "Aspartate_Aminotransferase", 
    "Alamine_Aminotransferase", edgecolor='w')
plt.subplots_adjust(top=0.9)
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", 
    data=liver_df, kind='reg')
# 由相关性关系图可以看出两种转氨酶之间具有一定的线性相关性

#%%------------------- 讨论磷酸酶和氨基转移酶的相关性 ---------------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.scatter, "Alkaline_Phosphotase", 
    "Alamine_Aminotransferase", edgecolor="w")
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", 
    data=liver_df, kind='reg')
# 由相关性图可以看出两者没有明显的相关性

#%%------------------- 讨论总蛋白和白蛋白之间的相关关系 -------------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.scatter, "Total_Protiens", "Albumin", edgecolor="w")
sns.jointplot("Total_Protiens", "Albumin",  data=liver_df, kind='reg')
# 由图表可以看出两个变量间存在较强的相关性，需要删除其中一个变量

#%%----------------- 讨论白蛋白和白蛋白-球蛋白结合率之间的关系 -------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.scatter, "Albumin", "Albumin_and_Globulin_Ratio", edgecolor="w")
sns.jointplot("Albumin", "Albumin_and_Globulin_Ratio", data=liver_df, kind='reg')
# 由图可以看出，两个变量间具有较好的相关性，需要删除其中一个变量

#%%---------------- 讨论白蛋白-球蛋白结合率与总蛋白量之间的关系 -------------------#
plt.figure()
graph = sns.FacetGrid(liver_df, col='Gender', row='Dataset', margin_titles=True)
graph.map(plt.scatter, "Albumin_and_Globulin_Ratio", 
    "Total_Protiens", edgecolor="w")
sns.jointplot("Albumin_and_Globulin_Ratio", 
    "Total_Protiens", data=liver_df, kind='reg')
# 由图可知，两个变量间没有明显的线性相关性

#%% 根据前面的调查情况进行下一步的数据处理
# 
# 由前面的关系图，我们发现了以下的具有相关关系的变量：
# Direct_Bilirubin & Total_Bilirubin
# Aspartate_Aminotransferase & Alamine_Aminotransferase
# Total_Protiens & Albumin
# Albumin_and_Globulin_Ratio & Albumin
# 在选择忽视其中的一些变量后，我们仅考虑以下变量：
# Total_Bilirubin
# Alamine_Aminotransferase
# Total_Protiens
# Albumin_and_Globulin_Ratio
# Albumin
# 
# 提取原始列表中性别一栏的性别字符串转化为两列由1、0表示的性别栏
gender_df = pd.get_dummies(liver_df['Gender'], prefix=None)
gender_df.head()
liver_df = pd.concat([liver_df, gender_df], axis=1) #组合原有数据和新建立的两列
liver_df.head()
liver_df.describe() #输出数据的详细统计结果

#-----------------------------------------------------------------------#
################################机器学习部分##############################
#-----------------------------------------------------------------------#
#%% 建立机器学习用的输入变量X
X = liver_df.drop(['Gender', 'Dataset'], axis=1)
X.head()

#%% 建立机器学习的标记变量Y
Y = liver_df.Dataset
Y.head()

#%% 计算并绘制相关性矩阵图，研究不同变量间的相关性情况
data_correlation = X.corr() #获取机器学习输入变量X的相关性矩阵
data_correlation #输出相关性矩阵
# 绘制相关系数矩阵热力图，显示不同变量间的相关性情况
plt.figure(figsize=(12, 12))
sns.heatmap(data_correlation, cbar=True, square=True, 
    annot=True, fmt='.2f', annot_kws={"size":15}, cmap='YlGnBu')
plt.title('Correlation between features')

#%% 进入机器学习前的头文件导入
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score #计算准确度函数
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
print('导入机器学习头文件成功！')

#%% 划分输入变量为训练数据和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0
)
print('数据分割成功，其中X_train:', X_train.shape, 'Y_train:', Y_train.shape, end='')
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)

#%% logistic回归分析模块
logreg = LogisticRegression().fit(X_train, Y_train) #创建并训练logistic回归模型
logpred = logreg.predict(X_test) #生成预测数据
logreg_score = logreg.score(X_train, Y_train) * 100  # 获取训练集得分百分比
logreg_score_test = logreg.score(X_test, Y_test) * 100  # 获取测试集得分百分比
print('logistic模型训练集正确率：%.2f%%' % logreg_score, end='')
print('，logistic模型测试集正确率：%.2f%%' % logreg_score_test)
print('logistic模型的准确率为：%.4f' % accuracy_score(Y_test, logpred))
print('logistic模型的模糊矩阵为：\n', confusion_matrix(Y_test, logpred))
print('logistic分类器分类结果为：\n', classification_report(Y_test, logpred))
plt.figure() #绘制模糊矩阵热力图
sns.heatmap(confusion_matrix(Y_test, logpred), annot=True, fmt='d')

#%% 高斯贝叶斯分类算法
gaussian = GaussianNB().fit(X_train, Y_train) #建立高斯贝叶斯分类器并进行训练
gauss_pred = gaussian.predict(X_test) #预测测试集数据
gauss_score = gaussian.score(X_train, Y_train) * 100 #训练集数据正确率
gauss_score_test = gaussian.score(X_test, Y_test) * 100 #测试集数据正确率
print('高斯贝叶斯分类器训练集正确率：%.2f%%' % gauss_score, end='')
print('，测试集正确率：%.2f%%' % gauss_score_test)
print('高斯贝叶斯分类器准确率为：%.4f' % accuracy_score(Y_test, gauss_pred))
print('高斯贝叶斯分类器模糊矩阵为：\n', confusion_matrix(Y_test, gauss_pred))
print('高斯贝叶斯分类器预估结果为：\n', classification_report(Y_test, gauss_pred))
plt.figure()
sns.heatmap(confusion_matrix(Y_test, gauss_pred), annot=True, fmt='d')

#%% 随机森林分类算法
random_forest = RandomForestClassifier(n_estimators=100).fit(X_train, Y_train)
rf_pred = random_forest.predict(X_test)
rf_score = random_forest.score(X_train, Y_train) * 100
rf_score_test = random_forest.score(X_test, Y_test) * 100
print('随机森林训练集正确率为：%.2f%%' % rf_score, end='')
print('，测试集正确率为：%.2f%%' % rf_score_test)
print('随机森林准确率为：%.4f' % accuracy_score(Y_test, rf_pred))
print('随机森林模糊矩阵为：\n', confusion_matrix(Y_test, rf_pred))
print('随机森林算法预估结果为：\n', classification_report(Y_test, rf_pred))
plt.figure()
sns.heatmap(confusion_matrix(Y_test, rf_pred), annot=True, fmt='d')

#%% 多元线性回归模型
linear = LinearRegression().fit(X_train, Y_train) #建立多元线性回归模型并训练
linear_pred = linear.predict(X_test)
linear_score = linear.score(X_train, Y_train) * 100
linear_score_test = linear.score(X_test, Y_test) * 100
print('多元线性回归模型训练集正确率为：%.2f%%' % linear_score, end='')
print('，测试集正确率为：%.2f%%' % linear_score_test)

#%% 三个模型的相互评估
models = pd.DataFrame({
    'Model' : ['LinearRegression', 'Logistic Regression', 
            'Gaussian Naive Bayes','Random Forest'], 
    'Score' : [linear_score, logreg_score, gauss_score, rf_score], 
    'Test Score' : [linear_score_test, logreg_score_test, 
            gauss_score_test, rf_score_test]
}) #建立三种模型评估结果得分的数据集
#将新生成的数据集按测试集得分降序排序
models.sort_values(by='Test Score', ascending=False)

#%% 使用递归特征消除（RFE）筛选重要特征并生成新的数据集
rfe = RFE(linear, n_features_to_select=3) #从多元线性回归模型中选择三个最重要的特征
rfe.fit(X, Y)
print(X.columns.values[rfe.ranking_ == 1]) #输出RFE筛选出的三个特征的名称
finX = liver_df[['Total_Protiens', 'Albumin', 'Female']] #由筛选的特征构造新变量
finX.head()

#%% 由RFE筛选后的新变量重新构造数据集并通过logistic回归模型重新训练
finX_train, finX_test, finY_train, finY_test = train_test_split(
    finX, Y, test_size=0.33, random_state=None
)
finlogreg = LogisticRegression().fit(finX_train, finY_train)
finlogpred = finlogreg.predict(finX_test) #生成预测数据
finlogreg_score = finlogreg.score(finX_train, finY_train) * 100
finlogreg_score_test = finlogreg.score(finX_test, finY_test) * 100
print('修正logistic模型训练集正确率：%.2f%%' % finlogreg_score, end='')
print('，测试集正确率：%.2f%%' % finlogreg_score_test)
print('修正logistic模型的准确率为：%.4f' % accuracy_score(finY_test, finlogpred))
print('修正logistic模型的模糊矩阵为：\n', confusion_matrix(finY_test, finlogpred))
print('修正logistic分类器分类结果为：\n', classification_report(finY_test, finlogpred))
plt.figure() #绘制模糊矩阵热力图
sns.heatmap(confusion_matrix(finY_test, finlogpred), annot=True, fmt='d')

# 参考网址：https://www.kaggle.com/sanjames/liver-patients-analysis-prediction-accuracy