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

# 3. 学习提升
- 需要内存：10GB
- 运行时间：Tesla V100 32G，预计90分钟


