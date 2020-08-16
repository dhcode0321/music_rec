# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification

#data_in = 'D:\\bd\\share_folder\\a9a'
data_in = './samples2.data'

def load_data():
    target_list = []
    fea_row_list = []
    fea_col_list = []
    data_list = []

    row_idx = 0
    max_col = 0

    with open(data_in, 'r') as fd:
        for line in fd:
            ss = line.strip().split(' ')
            label = ss[0]
            fea = ss[1:]

            target_list.append(int(label))

            for fea_score in fea:
                sss = fea_score.strip().split(':')
                if len(sss) != 2:
                    continue
                feature, score = sss

                fea_row_list.append(row_idx)
                fea_col_list.append(int(feature))
                data_list.append(float(score))
                if int(feature) > max_col:
                    max_col = int(feature)

            row_idx += 1

    row = np.array(fea_row_list)
    col = np.array(fea_col_list)
    data = np.array(data_list)

    fea_datasets = csr_matrix((data, (row, col)), shape=(row_idx, max_col + 1))

    x_train, x_test, y_train, y_test = train_test_split(fea_datasets, target_list, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

X_train,X_test,y_train,y_test = load_data()


# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 55,   # 叶子节点数
    'max_depth':7,
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.6, # 建树的特征选择比例
    'bagging_fraction': 0.6, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    #'min_data_in_leaf':20,
    #'min_sum_hessian_in_leaf':0.001

}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params,lgb_train,num_boost_round=999,valid_sets=lgb_eval,early_stopping_rounds=5)

print('Save model...')
# 保存模型到文件
gbm.save_model('model.txt')

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

#print(y_test)
#print(y_pred)


# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# # Visualising the Random Forest Regression results in higher resolution and smoother curve
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# X = np.arange(0, len(y_test), 1)
# plt.scatter(X,y_test, color = 'red')
# plt.scatter(X,y_pred, color = 'blue')
# plt.show()
# sys.exit()
#
# X_Grid = np.arange(0, len(y_test), 1)
# X_Grid = X_Grid.reshape((len(X_Grid), 1))
# plt.scatter(X,Y, color = 'red')
# plt.plot(X_Grid, regressor.predict(X_Grid), color = 'blue')
# plt.title('Random Forest Regression Results')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
