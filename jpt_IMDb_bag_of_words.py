#%% 导入头文件
import os 
import re
import numpy as np 
import pandas as pd 

#%% 处理原始的IMDb数据压缩生成CSV文件，无需再次运行！！
# labels = {'pos':1, 'neg':0}
# df = pd.DataFrame()
# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = 'D:/project/machine_learning_databases/aclImdb/%s/%s' % (s, l)
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), 'r', encoding='UTF-8') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], ignore_index=True)
# df.columns = ['review', 'sentiment']
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('D:/project/machine_learning_databases/IMDb_movie_data.csv', 
#     index=False)

#%% 读取测试生成的文件
df = pd.read_csv('D:/project/machine_learning_databases/IMDb_movie_data.csv')
df.head(3)

#%% 使用正则表达式处理文本数据
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    return text
preprocessor(df.loc[0, 'review'][-50:]) #测试处理结果

#%% 移除DataFrame中的所有电影评论信息
df['review'] = df['review'].apply(preprocessor)

#%% 使用简单的拆分空格方式标记文档
def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')

#%% 词干提取与停用词移除(需要先用nltk.download('stopwords')才能运行)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runners like running and thus they run') #测试词干提取函数
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs' 
    + ' a lot')[-10:] if w not in stop]

#%% 划分训练数据和测试数据
X_train = df.loc[:25000, 'review'].values
Y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values

#%% 使用网格搜索5折分层交叉验证逻辑斯蒂回归模型最佳参数组合
# 由于无法使用多核运行，加上数据量太大，训练估计要40分钟时间，因此放弃运行
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, 
    lowercase=False, preprocessor=None)
param_grid = [
    {
        'vect__ngram_range':[(1, 1)], 
        'vect__stop_words':[stop, None], 
        'vect__tokenizer':[tokenizer_porter, tokenizer], 
        'clf__penalty':['l1', 'l2'], 
        'clf__C':[1.0, 10.0, 100.0]
    }, 
    {
        'vect__ngram_range':[(1, 1)], 
        'vect__stop_words':[stop, None], 
        'vect__tokenizer':[tokenizer_porter, tokenizer], 
        'vect__use_idf':[False], 
        'vect__norm':[None], 
        'clf__penalty':['l1', 'l2'], 
        'clf__C':[1.0, 10.0, 100.0]
    }
]
lr_tfidf = Pipeline([
    ('vect', tfidf), 
    ('clf', LogisticRegression(random_state=0))
    ])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', 
    cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(X_train, Y_train)
print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, Y_test))