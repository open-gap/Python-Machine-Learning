#%% 导入头文件
import re 
import numpy as np 
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

#%% 定义文本清洗函数
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', 
        text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

#%% 定义生成器函数，每次读取并返回一个文档的内容
def stream_docs(path):
    with open(path, 'r', encoding='UTF-8') as csv:
        next(csv) # 跳过文件头
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
# 测试生成器是否正常
next(stream_docs(path = \
    'D:/project/machine_learning_databases/IMDb_movie_data.csv'))

#%% 定义返回指定数量文档内容的函数
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

#%% 利用哈希向量处理器与随机梯度下降分类器构建模型
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
    preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path = 
    'D:/project/machine_learning_databases/IMDb_movie_data.csv')
classes = np.array([0, 1])
batch_num = 45 # 文档批次
for i in range(batch_num):
    X_train, Y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, Y_train, classes=classes)
    print('%d/%d batch done.' % (i + 1, batch_num))

X_test, Y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Test Accuracy: %.3f' % clf.score(X_test, Y_test))
# 可利用测试数据进一步学习
# clf = clf.partial_fit(X_test, Y_test)

#%% 模型的本地保存
import os 
import pickle
dest = os.path.join('D:/project/machine_learning_databases/', 
    'movieclassifier/pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), 
    protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), 
    protocol=4)