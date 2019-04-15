#%% 导入头文件
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

#%% 读取CSV文件数据
wine_df = pd.read_csv(
    r'D:/project/machine_learning_databases/winequality-red.csv')
wine_df.tail(7)

#%% 输出基本情况
print(wine_df.info()) #输出读取数据的基本情况
print(wine_df.columns) #输出数据集的列名称
print(wine_df.isnull().sum()) #输出缺失数据情况

#%% 输入数据的基本分析
sns.countplot(x='quality', data=wine_df, label='Count') #评分情况总览
data_correlation = wine_df.iloc[:, :-1].corr() #计算葡萄酒参数的相关系数矩阵
plt.figure(figsize=(12, 12)) #绘制相关系数矩阵热力图，显示不同变量间的相关性情况
sns.heatmap(data_correlation, cbar=True, square=True, 
    annot=True, fmt='.2f', annot_kws={"size":15}, cmap='YlGnBu')
plt.title('Correlation between features')

#%% 去除相关系数绝对值大于0.55两个变量中的一个以减少相关性干扰
wine_df.drop(labels=['fixed acidity', 'free sulfur dioxide'],
    axis=1, inplace=True)
data_correlation = wine_df.iloc[:, :-1].corr() #计算葡萄酒参数的相关系数矩阵
plt.figure(figsize=(12, 12)) #绘制相关系数矩阵热力图，显示不同变量间的相关性情况
sns.heatmap(data_correlation, cbar=True, square=True, 
    annot=True, fmt='.2f', annot_kws={"size":15}, cmap='YlGnBu')
plt.title('Correlation between features')

#%% 制作机器学习的数据集
X, Y = wine_df.iloc[:, :-1], wine_df.iloc[:, -1] #数据获取与切分
print('X.shape: ' + str(X.shape), '\nY.shape: ' + str(Y.shape))
X_train, X_test, Y_train, Y_test = train_test_split( 
    X, Y, test_size=0.33, random_state=0, shuffle=True
)
stdsc = StandardScaler() #数据标准化
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)
X_train, X_test = X_train.T, X_test.T #修改数据形状便于后期编程
Y_train, Y_test = Y_train[:, np.newaxis].T, Y_test[:, np.newaxis].T #避免维度问题
print('X_train.shape: ' + str(X_train.shape))
print('Y_train.shape: ' + str(Y_train.shape))
print('X_test.shape: ' + str(X_test.shape))
print('Y_test.shape: ' + str(Y_test.shape))

#%% 参数初始化 #################################################################
def parameters_initialize(num_in, nn_layers_list):
    parameters = {}
    last_count = num_in
    for i, n in enumerate(nn_layers_list):
        parameters['W' + str(i+1)] = np.random.randn(n, last_count) * 0.1
        parameters['b' + str(i+1)] = np.random.randn(n, 1) * 0.1
        parameters['Z' + str(i+1)] = np.zeros((n, 1))
        parameters['A' + str(i+1)] = np.zeros((n, 1))
        last_count = n
    
    return parameters

#%% 测试参数初始化函数
parameters = parameters_initialize(10, [8, 6, 3, 1])
print('W1.shape = ' + str(parameters['W1'].shape))
print('b1.shape = ' + str(parameters['b1'].shape))
print('Z1.shape = ' + str(parameters['Z1'].shape))
print('A1.shape = ' + str(parameters['A1'].shape))
print('W2.shape = ' + str(parameters['W2'].shape))
print('b2.shape = ' + str(parameters['b2'].shape))
print('Z2.shape = ' + str(parameters['Z2'].shape))
print('A2.shape = ' + str(parameters['A2'].shape))
print('Wn.shape = ' + str(parameters['W4'].shape))
print('bn.shape = ' + str(parameters['b4'].shape))
print('Zn.shape = ' + str(parameters['Z4'].shape))
print('An.shape = ' + str(parameters['A4'].shape))

#%% 前向传递的计算
def relu(X_array):
    return np.where(X_array > 0, X_array, 0)

def sigmoid(X_array):
    return 1.0/(1.0 + np.exp(-X_array))

# def softmax(Y_array):
#     assert Y_array.shape[0] > 1, 'Input shape rows error'
#     assert Y_array.shape[1] == 1, 'Input shape columns error'
#     temp_Y = np.exp(Y_array)
#     return temp_Y/np.sum(temp_Y)

def forward_propagation(X, parameters):
    assert(len(parameters) % 4 == 0), 'Parameters is wrong'
    layers = len(parameters) // 4
    A_last = X
    for l in range(layers-1):
        parameters['Z' + str(l+1)] = \
            parameters['W' + str(l+1)].dot(A_last) + parameters['b' + str(l+1)]
        parameters['A' + str(l+1)] = relu(parameters['Z' + str(l+1)])
        A_last = parameters['A' + str(l+1)]

    parameters['Z' + str(layers)] = \
        parameters['W' + str(layers)].dot(A_last) + parameters['b' + str(layers)]
    parameters['A' + str(layers)] = sigmoid(parameters['Z' + str(layers)])

    return parameters['A' + str(layers)], parameters

#%% 测试前向传递函数
parameters = {
    'W1' : np.array([[1,-2,3,-4,5],[-6,7,-8,9,-10],[11,-12,13,-14,15]])*0.1, 
    'b1' : np.array([[0.1],[-0.2],[0.3]]), 
    'W2' : np.array([[-1,-2,3],[-4,5,-6]])*0.1, 
    'b2' : np.array([[0.4],[-0.5]]), 
    'W3' : np.array([2,-4])*0.1, 
    'b3' : np.array([0.07]).reshape(1,1)
    }
A_last, parameters = forward_propagation(np.ones((5, 1)), parameters)
print('A_last = ', A_last.squeeze())
print('parameters : ', parameters)

#%% 代价函数的计算
def compute_cost(Y_hat, Y):
    assert(Y.shape[1] >= 1), 'Y shape is wrong'
    assert(Y_hat.shape[1] >= 1), 'Y_hat shape is wrong'
    # assert(len(parameters) % 4 == 0), 'Input parameters is wrong'
    J = np.sum(- Y*np.log(Y_hat) - (1-Y)*np.log(1-Y_hat)) / Y.shape[1]
    # lambd = 0.01
    # regulation = 0
    # for l in range(len(parameters) // 4):
    #     regulation += np.linalg.norm(parameters['W' + str(l+1)])
    # return (np.sum(J) + lambd * regulation / 2.0) / m
    return J

#%% 测试代价函数
cost = compute_cost(
    np.array([[0.7,0.001],[0.35,0.998]]), np.array([[1,0],[0,1]]))
# cost = compute_cost(
#     np.array([[0.7],[0.35]]), np.array([[1],[0]]), parameters
# )
print('cost = %.7f' % cost)

#%% 梯度下降计算函数
def gradient_decent(X, Y, parameters):
    assert(len(parameters) % 4 == 0), 'Parameters is wrong'
    assert(X.shape[1] == Y.shape[1]), 'Y.shape columns are wrong'
    layers = len(parameters) // 4
    m = Y.shape[1]
    grad = {}

    grad['dA' + str(layers)] = \
        -np.divide(Y, parameters['A' + str(layers)]) \
        +np.divide(1-Y, 1-parameters['A' + str(layers)])
    dZ = grad['dA' + str(layers)] * sigmoid(parameters['Z' + str(layers)]) \
        * (1 - sigmoid(parameters['Z' + str(layers)]))
    if(layers <= 1):
        grad['dW' + str(layers)] = dZ.dot(X.T) / m
        grad['db' + str(layers)] = np.sum(dZ, axis=1, keepdims=True) / m
        return grad
    
    grad['dW' + str(layers)] = dZ.dot(parameters['A' + str(layers-1)].T) / m
    grad['db' + str(layers)] = np.sum(dZ, axis=1, keepdims=True) / m
    grad['dA' + str(layers - 1)] = parameters['W' + str(layers)].T.dot(dZ)
    for l in reversed(range(layers - 1)):
        dZ = grad['dA' + str(l+1)] * (parameters['Z' + str(l+1)] > 0)
        if l > 0:
            grad['dW' + str(l+1)] = dZ.dot(parameters['A' + str(l)].T) / m
            grad['db' + str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / m
            grad['dA' + str(l)] = parameters['W' + str(l+1)].T.dot(dZ)
        else:
            grad['dW' + str(l+1)] = dZ.dot(X.T) / m
            grad['db' + str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / m

    return grad

#%% 测试梯度下降函数
np.random.seed(0)
parameters = parameters_initialize(5, [3,1])
X = np.ones((5, 1))
X = np.hstack((X, np.zeros((5,1))))
Y_hat, parameters = forward_propagation(X, parameters)
print("Y_hat = ", Y_hat)
print(gradient_decent(X, np.array([[1,0]]), parameters))

#%% 参数更新函数
def update_parameters(grad, parameters, t, v, s, learning_rate=0.01, 
    method='Adam', beta1=0.9, beta2=0.999, epsilon=1e-8):

    assert(len(parameters) % 4 == 0), 'Parameters is wrong'
    layers = len(parameters) // 4
    for l in range(layers):
        v['dW' + str(l+1)] = beta1 * v['dW' + str(l+1)] + \
            (1 - beta1) * grad['dW' + str(l+1)]
        v['db' + str(l+1)] = beta1 * v['db' + str(l+1)] + \
            (1 - beta1) * grad['db' + str(l+1)]
        s['dW' + str(l+1)] = beta2 * s['dW' + str(l+1)] + \
            (1 - beta1) * np.square(grad['dW' + str(l+1)])
        s['db' + str(l+1)] = beta2 * s['db' + str(l+1)] + \
            (1 - beta1) * np.square(grad['db' + str(l+1)])
    
        if(method == 'Adam'):
            adam_w = v['dW' + str(l+1)]/(1 - beta1**t) / \
                (np.sqrt(s['dW' + str(l+1)]/(1 - beta2**t)) + epsilon)
            parameters['W' + str(l+1)] -= learning_rate * adam_w
            adam_b = v['db' + str(l+1)]/(1 - beta1**t) / \
                (np.sqrt(s['db' + str(l+1)]/(1 - beta2**t)) + epsilon)
            parameters['b' + str(l+1)] -= learning_rate * adam_b
        elif(method == 'Momentum'):
            momt_w = v['dW' + str(l+1)]/(1 - beta1**t)
            parameters['W' + str(l+1)] -= learning_rate * momt_w
            momt_b = v['db' + str(l+1)]/(1 - beta1**t)
            parameters['b' + str(l+1)] -= learning_rate * momt_b
        elif(method == 'RMSprop'):
            rmsp_w = np.sqrt(s['dW' + str(l+1)]/(1 - beta2**t)) + epsilon
            parameters['W' + str(l+1)] -= \
                learning_rate * grad['dW' + str(l+1)] / rmsp_w
            rmsp_b = np.sqrt(s['db' + str(l+1)]/(1 - beta2**t)) + epsilon
            parameters['b' + str(l+1)] -= \
                learning_rate * grad['db' + str(l+1)] / rmsp_b
        else:
            parameters['W' + str(l+1)] -= learning_rate * grad['dW' + str(l+1)]
            parameters['b' + str(l+1)] -= learning_rate * grad['db' + str(l+1)]

    return parameters, v, s

#%% 组合不同函数，增加mini-batch更新功能
def nn_model(X, Y, nn_layers_list, minibatch=None, learning_rate=0.01, 
    method='Adam', drop_rate=1.0, epochs=1000, print_process=True):

    assert(Y.shape[0] == nn_layers_list[-1]), 'Output dims are wrong'
    layers = len(nn_layers_list)
    assert(layers >= 2), 'Neural Network has too few layers'

    num_in, m = X.shape
    parameters = parameters_initialize(num_in, nn_layers_list)
    costs = []
    for num in range(epochs):
        choice = np.arange(m)
        np.random.shuffle(choice)
        temp_parameters = parameters.copy()
        v, s = {}, {}
        for l in range(len(nn_layers_list)):
            v['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
            v['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
            s['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
            s['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        if minibatch:
            assert(m > minibatch), 'Minibatch sets too large'
            assert(type(minibatch) == np.int), 'Minibatch type isnt int'
            iter = m // minibatch
            for i in range(iter):
                temp_X = X[:, choice[i*minibatch : (i+1)*minibatch]]
                temp_Y = Y[:, choice[i*minibatch : (i+1)*minibatch]]
                if(drop_rate < 1.0):
                    for j in range(len(nn_layers_list)):
                        sel = np.random.random(parameters['W' + str(j+1)].shape)
                        temp_parameters['W' + str(j+1)] *= (sel < drop_rate)
                        sel = np.random.random(parameters['b' + str(j+1)].shape)
                        temp_parameters['b' + str(j+1)] *= (sel < drop_rate)
                Y_hat, temp_parameters = forward_propagation(temp_X, temp_parameters)
                cost = compute_cost(Y_hat, temp_Y)
                grad = gradient_decent(temp_X, temp_Y, temp_parameters)
                parameters, v, s = update_parameters(grad, temp_parameters, 
                    num + 1, v, s, learning_rate, method)
            costs.append(cost)
        else:
            if(drop_rate < 1.0):
                for j in range(len(nn_layers_list)):
                    sel = np.random.random(parameters['W' + str(j+1)].shape)
                    temp_parameters['W' + str(j+1)] *= (sel < drop_rate)
                    sel = np.random.random(parameters['b' + str(j+1)].shape)
                    temp_parameters['b' + str(j+1)] *= (sel < drop_rate)
            Y_hat, temp_parameters = forward_propagation(X, temp_parameters)
            cost = compute_cost(Y_hat, Y)
            costs.append(cost)
            grad = gradient_decent(X, Y, temp_parameters)
            parameters, v, s = update_parameters(grad, temp_parameters, 
                num + 1, v, s, learning_rate, method)

        # if print_process:
        if print_process and (num % 10 == 0):
            print('%d/%d epochs cost is %.5f' % (num + 1, epochs, cost))
    
    return parameters, costs

#%% 使用神经网络函数
X = np.arange(400).reshape(2, 200)
# Y = np.zeros((1, 50))
# Y[0, [i for i in range(Y.shape[1]) if (i % 2 == 0)]] = 1
Y = np.where(np.sin(np.sum(X, axis=0, keepdims=True)) > 0, 1, 0)
epoch = 1000
parameters, costs = nn_model(X, Y, [2, 1],
    minibatch=None, learning_rate=0.003, method='Adam', 
    drop_rate=1.0, epochs=epoch, print_process=False)
print('Final cost is', costs[-1])
plt.plot(range(epoch), costs)
X_test = np.arange(200, 280).reshape(2, 40)
Y_test = np.where(np.sin(np.sum(X_test, axis=0, keepdims=True)) > 0, 1, 0)
Y_hat, _ = forward_propagation(X_test, parameters)
Y_hat = np.where(Y_hat > 0.5, 1, 0)
accuracy = np.sum(np.equal(Y_hat, Y_test)) / X_test.shape[1]
print('Nerual Network accuracy is %.5f' % accuracy)

#%% 测试单个函数
X = np.linspace(-10, 10, num=100)
Y = sigmoid(X)
plt.plot(X, Y)