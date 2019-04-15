#%% 模型的本地保存，其中stop为停止词向量，clf为分类器参数
# import os 
# import pickle
# dest = os.path.join('D:/project/machine_learning_databases/', 
#     'movieclassifier/pkl_objects')
# if not os.path.exists(dest):
#     os.makedirs(dest)
# pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), 
#     protocol=4)
# pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), 
#     protocol=4)

#%% 导入头文件
import os 
import re 
import pickle
import numpy as np 
from sklearn.feature_extraction.text import HashingVectorizer

#%% 读取保存的模型数据
stop = pickle.load(
    open(os.path.join(
    'D:/project/machine_learning_databases/movieclassifier/pkl_objects/', 
    'stopwords.pkl'), 'rb')
)
clf = pickle.load(
    open(os.path.join(
        'D:/project/machine_learning_databases/movieclassifier/pkl_objects/', 
        'classifier.pkl'), 'rb')
)

#%% 定义模型需要用到的函数和哈希向量模型
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', 
        text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
    preprocessor=None, tokenizer=tokenizer)

#%% 使用模型进行预测
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.3f%%' % (label[clf.predict(X)[0]], 
    np.max(clf.predict_proba(X)) * 100))