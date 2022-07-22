# 赛题
基于论文摘要的文本分类与查询性问答

## 实践任务
要求选手构建模型，让机器通过对论文摘要等信息的理解，划分论文类别。

## 实践思路
是一个典型的文本分类任务。由于文本数据是典型的非结构化数据，此类实践的处理通常涉及到 特征提取 和 分类模型 两部分。
常见的思路有两种：基于机器学习的思路和基于深度学习的思路。

本代码尝试基于机器学习的思路：TF-IDF + 机器学习分类器，其中分类器选择SGD线性分类器。
SGD是线性分类器的一种，可以理解为逻辑回归+随机梯度下降，适合处理文本TF-IDF编码后的稀疏场景。

## 代码实现
```
#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas
#---------------------------------------------------
#导入库
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

#----------------数据探索----------------
#数据预处理
#加载训练集
train_df = pd.read_csv('./基于论文摘要的文本分类与查询性问答公开数据/train.csv', sep=',')
#加载测试集
test_df = pd.read_csv('./基于论文摘要的文本分类与查询性问答公开数据/test.csv', sep=',')

#EDA数据探索性分析
train_df.head()

test_df.head()

#----------------特征工程----------------
#将Topic(Label)编码
train_df['Topic(Label)'], lbl = pd.factorize(train_df['Topic(Label)'])

#将论文的标题与摘要组合为 text 特征
train_df['Title'] = train_df['Title'].apply(lambda x: x.strip())
train_df['Abstract'] = train_df['Abstract'].fillna('').apply(lambda x: x.strip())
train_df['text'] = train_df['Title'] + ' ' + train_df['Abstract']
train_df['text'] = train_df['text'].str.lower()

test_df['Title'] = test_df['Title'].apply(lambda x: x.strip())
test_df['Abstract'] = test_df['Abstract'].fillna('').apply(lambda x: x.strip())
test_df['text'] = test_df['Title'] + ' ' + test_df['Abstract']
test_df['text'] = test_df['text'].str.lower()

#使用tfidf算法做文本特征提取
tfidf = TfidfVectorizer(max_features=2500)

#----------------模型训练----------------

train_tfidf = tfidf.fit_transform(train_df['text'])
clf = SGDClassifier()
cross_val_score(clf, train_tfidf, train_df['Topic(Label)'], cv=5)

test_tfidf = tfidf.transform(test_df['text'])
clf = SGDClassifier()
clf.fit(train_tfidf, train_df['Topic(Label)'])
test_df['Topic(Label)'] = clf.predict(test_tfidf)

#----------------结果输出----------------
test_df['Topic(Label)'] = test_df['Topic(Label)'].apply(lambda x: lbl[x])
test_df[['Topic(Label)']].to_csv('submit.csv', index=None)
```

## 3. 学习提升
上述代码详细讲解了基于机器学习的思路，若想进阶实践，可考虑尝试基于深度学习来进行实践，提供以下几种常见解题思路供大家参考：

- 思路1：FastText：FastText是入门款的词向量，利用Facebook提供的FastText工具，可以快速构建出分类器。
- 思路2：WordVec + 深度学习分类器：WordVec是进阶款的词向量，并通过构建深度学习分类完成分类。深度学习分类的网络结构可以选择TextCNN、TextRNN或者BiLSTM。
- 思路3：Bert词向量：Bert是高配款的词向量，具有强大的建模学习能力。
