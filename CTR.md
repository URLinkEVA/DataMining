# 赛题
广告-信息流跨域ctr预估

[链接](https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html#/preliminary/info/006/introduction)

# 赛题说明
本赛题提供7天数据用于训练，1天数据用于测试，数据包括目标域（广告域）用户行为日志，用户基本信息，广告素材信息，源域（信息流域）用户行为数据，源域（信息流域）物品基本信息等。希望选手基于给出的数据，识别并生成源域能反映用户兴趣，并能应用于目标域的用户行为特征表示，基于用户行为序列信息，进行源域和目标域的联合建模，预测用户在广告域的点击率。所提供的数据经过脱敏处理，保证数据安全。

# 实践思路
本次比赛是一个经典点击率预估（CTR）的数据挖掘赛，需要选手通过训练集数据构建模型，然后对验证集数据进行预测，预测结果进行提交。

本题的任务是构建一种模型，根据用户的测试数据来预测这个用户是否点击广告。这种类型的任务是典型的二分类问题，模型的预测输出为 0 或 1 （点击：1，未点击：0）

机器学习中，关于分类任务我们一般会想到逻辑回归、决策树等算法，在这个 Baseline 中，我们尝试使用逻辑回归来构建我们的模型。

# 代码实现
- 需要内存：1GB
- 运行时间：5分钟

```
#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas

#---------------------------------------------------
#导入库
import pandas as pd

#----------------数据探索----------------
# 只使用目标域用户行为数据
train_ads = pd.read_csv('./train/train_data_ads.csv',
    usecols=['log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'device_name',
            'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])

test_ads = pd.read_csv('./test/test_data_ads.csv',
    usecols=['log_id', 'user_id', 'age', 'gender', 'residence', 'device_name',
    'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])
    
# 数据集采样  
train_ads = pd.concat([
    train_ads[train_ads['label'] == 0].sample(70000),
    train_ads[train_ads['label'] == 1].sample(10000),
])


#----------------模型训练----------------
# 加载训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(
    train_ads.drop(['log_id', 'label', 'user_id'], axis=1),
    train_ads['label']
)


#----------------结果输出----------------
# 模型预测与生成结果文件
test_ads['pctr'] = clf.predict_proba(
    test_ads.drop(['log_id', 'user_id'], axis=1),
)[:, 1]
test_ads[['log_id', 'pctr']].to_csv('submission.csv',index=None)#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas

#---------------------------------------------------
#导入库
import pandas as pd

#----------------数据探索----------------
# 只使用目标域用户行为数据
train_ads = pd.read_csv('./train/train_data_ads.csv',
    usecols=['log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'device_name',
            'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])

test_ads = pd.read_csv('./test/test_data_ads.csv',
    usecols=['log_id', 'user_id', 'age', 'gender', 'residence', 'device_name',
    'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])
    
# 数据集采样  
train_ads = pd.concat([
    train_ads[train_ads['label'] == 0].sample(70000),
    train_ads[train_ads['label'] == 1].sample(10000),
])


#----------------模型训练----------------
# 加载训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(
    train_ads.drop(['log_id', 'label', 'user_id'], axis=1),
    train_ads['label']
)


#----------------结果输出----------------
# 模型预测与生成结果文件
test_ads['pctr'] = clf.predict_proba(
    test_ads.drop(['log_id', 'user_id'], axis=1),
)[:, 1]
test_ads[['log_id', 'pctr']].to_csv('submission.csv',index=None)
```

# 学习提升
- 需要内存：10GB
- 运行时间：Tesla V100 32G，预计90分钟

## 导入模块
```
#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas
#!pip install catboost
#---------------------------------------------------
#导入库
#----------------数据探索----------------
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from tqdm import *
#----------------核心模型----------------
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
#----------------交叉验证----------------
from sklearn.model_selection import StratifiedKFold, KFold
#----------------评估指标----------------
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
#----------------忽略报警----------------
import warnings
warnings.filterwarnings('ignore')
```
## 数据预处理
```
# 读取训练数据和测试数据
train_data_ads = pd.read_csv('./2022_3_data/train/train_data_ads.csv')
train_data_feeds = pd.read_csv('./2022_3_data/train/train_data_feeds.csv')

test_data_ads = pd.read_csv('./2022_3_data/test/test_data_ads.csv')
test_data_feeds = pd.read_csv('./2022_3_data/test/test_data_feeds.csv')

# 合并数据
# 合并数据
train_data_ads['istest'] = 0
test_data_ads['istest'] = 1
data_ads = pd.concat([train_data_ads, test_data_ads], axis=0, ignore_index=True)

train_data_feeds['istest'] = 0
test_data_feeds['istest'] = 1
data_feeds = pd.concat([train_data_feeds, test_data_feeds], axis=0, ignore_index=True)

del train_data_ads, test_data_ads, train_data_feeds, test_data_feeds
gc.collect()
```
## 特征工程
### 自然数编码
```
def label_encode(series, series2):
    unique = list(series.unique())
    return series2.map(dict(zip(
        unique, range(series.nunique())
    )))

for col in ['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003','ad_close_list_v001','ad_close_list_v002','ad_close_list_v003','u_newsCatInterestsST']:
    data_ads[col] = label_encode(data_ads[col], data_ads[col])
```

### 特征提取
```
# data_feeds特征构建
cols = [f for f in data_feeds.columns if f not in ['label','istest','u_userId']]
for col in tqdm(cols):
    tmp = data_feeds.groupby(['u_userId'])[col].nunique().reset_index()
    tmp.columns = ['user_id', col+'_feeds_nuni']
    data_ads = data_ads.merge(tmp, on='user_id', how='left')

cols = [f for f in data_feeds.columns if f not in ['istest','u_userId','u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news','i_docId','i_s_sourceId','i_entities']]
for col in tqdm(cols):
    tmp = data_feeds.groupby(['u_userId'])[col].mean().reset_index()
    tmp.columns = ['user_id', col+'_feeds_mean']
    data_ads = data_ads.merge(tmp, on='user_id', how='left')
```
特征提取部分围绕着data_feeds进行构建（添加源域信息），主要是nunique属性数统计和mean均值统计。由于是baseline方案，所以整体的提取比较粗暴，大家还是有很多的优化空间。

### 内存压缩
```
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
# 压缩使用内存
data_ads = reduce_mem_usage(data_ads)
# Mem. usage decreased to 2351.47 Mb (69.3% reduction)
```
由于数据比较大，所以合理的压缩内存节省空间尤为的重要，使用reduce_mem_usage函数可以压缩近70%的内存占有。

# 训练集测试集划分
```
# 划分训练集和测试集
cols = [f for f in data_ads.columns if f not in ['label','istest']]
x_train = data_ads[data_ads.istest==0][cols]
x_test = data_ads[data_ads.istest==1][cols]

y_train = data_ads[data_ads.istest==0]['label']

del data_ads, data_feeds
gc.collect()
```

# 训练模型
```
def cv_model(clf, train_x, train_y, test_x, clf_name, seed=2022):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} {}************************************'.format(str(i+1), str(seed)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
               
        params = {'learning_rate': 0.3, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type':'Bernoulli','random_seed':seed,
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=20000, **params, eval_metric='AUC')
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  metric_period=200,
                  cat_features=[], 
                  use_best_model=True, 
                  verbose=1)

        val_pred  = model.predict_proba(val_x)[:,1]
        test_pred = model.predict_proba(test_x)[:,1]
            
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

cat_train, cat_test = cv_model(CatBoostClassifier, x_train, y_train, x_test, "cat")
```
# 结果保存
```
x_test['pctr'] = cat_test
x_test[['log_id','pctr']].to_csv('submission.csv', index=False)
```

# 接下来方向
- 继续尝试不同的预测模型或特征工程来提升模型预测的准确度
- 尝试模型融合等策略
- 查阅广告信息流跨域ctr预估预测相关资料，获取其他模型构建方法
