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
from sklearn.model_selection import GridSearchCV

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
#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 30,   # 叶子节点数
    #'max_depth':7,
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

#gbm = lgb.train(params,lgb_train,num_boost_round=60,valid_sets=lgb_eval,early_stopping_rounds=5)

#data_train = lgb.Dataset(df_train, y_train, silent=True)

params_test3={ 'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,learning_rate=0.05, max_depth=7, metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)



# 评估模型

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
