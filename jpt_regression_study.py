#%% 导入头文件
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#%% 数据集文件的读取与显示
df = pd.read_csv(
    'D:/project/machine_learning_databases/Boston Housing/housing.data', 
    sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATO', 'B', 'LSTAT', 'MEDV']
df.head()

#%% 探索性数据分析(Exploratory Data Analysis, EDA)
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=3)
plt.show()
# sns.reset_orig() # 由于seaborn会覆盖matplotlib风格，可使用该语句还原

#%% 计算相关系数矩阵并绘制热力图
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
    annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

#%% 定义线性回归模型
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        self.w_ = np.zeros((1 + X.shape[1], 1))
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (Y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / X.shape[0]
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

#%% 使用房间数量(RM)预测房屋价格(MEDV)并绘图
X = df['RM'].values.reshape(-1, 1)
Y = df['MEDV'].values.reshape(-1, 1)
X_std = StandardScaler().fit_transform(X)
Y_std = StandardScaler().fit_transform(Y)
lr = LinearRegressionGD().fit(X_std, Y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

#%% 绘制房间数与房屋价格之间的关系图
def lin_regplot(X, Y, model):
    plt.scatter(X, Y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X_std, Y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

#%% 将预测结果反向缩放到原始区间
num_rooms_std = StandardScaler().fit(X).transform(np.array([5.0]).reshape(1, 1))
price_std = lr.predict(num_rooms_std)
print('Price in $1000\'s: %.3f' % StandardScaler().fit(Y).inverse_transform(price_std))
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])

#%% 使用sklearn库函数实现高效运算
slr = LinearRegression().fit(X, Y)
print('Slope: %.3f' % slr.coef_[0]) # 斜率
print('Intercept: %.3f' % slr.intercept_) # 截距
lin_regplot(X, Y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')

#%% 使用随机抽样一致性(RANdomSAmple Consensus, RANSAC)算法进行回归拟合
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, 
    residual_threshold=5.0, random_state=0)
ransac.fit(X, Y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_Y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], Y[inlier_mask], c='blue', marker='o', 
    label='Inliers')
plt.scatter(X[outlier_mask], Y[outlier_mask], c='lightgreen', marker='s', 
    label='Outliers')
plt.plot(line_X, line_Y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()
print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

#%% 划分训练集与数据集并使用所有特征量进行学习并绘制残差图
X = df.iloc[:, :-1].values
Y = df['MEDV'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0
)
slr = LinearRegression().fit(X_train, Y_train)
Y_train_pred = slr.predict(X_train)
Y_test_pred = slr.predict(X_test)
plt.scatter(Y_train_pred, Y_train_pred - Y_train, c='blue', marker='o', 
    label='Training data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test, c='lightgreen', marker='s', 
    label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
plt.xlim([-10, 50])
plt.show()

#%% 评估模型性能的一些方法
from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(Y_train, Y_train_pred), 
    mean_squared_error(Y_test, Y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(Y_train, Y_train_pred), r2_score(Y_test, Y_test_pred)))

#%% 其他回归
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0) # 岭回归
lasso = Lasso(alpha=1.0) # 最小绝对收缩及算子选择回归
elasnet = ElasticNet(alpha=1.0, l1_ratio=0.5) # 弹性网络回归

#%% 对比简单线性回归和多项式回归
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 
    396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
Y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 
    360.8, 368.0, 391.2, 390.8])[:, np.newaxis]
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

lr.fit(X, Y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
Y_lin_fit = lr.predict(X_fit)
pr.fit(X_quad, Y)
Y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
plt.scatter(X, Y, label='Training Points')
plt.plot(X_fit, Y_lin_fit, label='Linear Fit', linestyle='--')
plt.plot(X_fit, Y_quad_fit, label='Quadratic Fit')
plt.show()

Y_lin_pred = lr.predict(X)
Y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (
    mean_squared_error(Y, Y_lin_pred), mean_squared_error(Y, Y_quad_pred)
    ))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
    r2_score(Y, Y_lin_pred), r2_score(Y, Y_quad_pred)
    ))

#%% 转化输入输出变量，化为线性关系再拟合的方式
X_log = np.log(df['LSTAT'].values).reshape(-1, 1)
Y_sqrt = np.sqrt(df['MEDV'].values).reshape(-1, 1)
X_fit = np.arange(X_log.min() - 1, X_log.max() + 1)[:, np.newaxis]
regr = LinearRegression().fit(X_log, Y_sqrt)
Y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(Y_sqrt, regr.predict(X_log))
plt.scatter(X_log, Y_sqrt, label='Training Data', color='lightgray')
plt.plot(X_fit, Y_lin_fit, label='Linear (d=1), $R^2=%.3f$' % linear_r2, 
    color='blue', lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
plt.legend(loc='lower left')
plt.show()

#%% 使用决策树回归预测样本关系
from sklearn.tree import DecisionTreeRegressor
X = df['LSTAT'].values.reshape(-1, 1)
Y = df['MEDV'].values.reshape(-1, 1)
tree = DecisionTreeRegressor(max_depth=3).fit(X, Y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], Y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in 1000\'s [MEDV]')
plt.show()

#%% 随机森林回归
from sklearn.ensemble import RandomForestRegressor
X = df.iloc[:, :-1].values
Y = df['MEDV'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=1
)
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', 
    random_state=1, n_jobs=-1)
forest.fit(X_train, Y_train)
Y_train_pred = forest.predict(X_train)
Y_test_pred = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(Y_train, Y_train_pred), 
    mean_squared_error(Y_test, Y_test_pred)
))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(Y_train, Y_train_pred), r2_score(Y_test, Y_test_pred)
))
plt.scatter(Y_test_pred, Y_test_pred - Y_test, c='lightgreen', marker='s', 
    s=35, alpha=0.7, label='Test data')
plt.scatter(Y_train_pred, Y_train_pred - Y_train, c='black', marker='o', 
    s=35, alpha=0.5, label='Training data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
plt.xlim([-10, 50])
plt.show()