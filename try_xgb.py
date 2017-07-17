import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.metrics as metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

def print_result(train_type, y_test, y_hat, train_time):

    acc = accuracy_score(y_test, y_hat)
    pre = precision_score(y_test, y_hat)
    rec = recall_score(y_test, y_hat)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat)
    auc = metrics.auc(fpr, tpr)
    print('模型为: ',train_type)
    print('准确率：%.4f' % acc)
    print('精确率：%.4f' % pre)
    print('召回率：%.4f' % rec)
    print('AUC：%.4f' % auc)
    print('耗时: %.3f秒' % train_time)

if __name__ == "__main__":

    data_part1 = pd.read_csv('data/train_data.csv')
    data_part1.drop_duplicates(inplace=True)
    data_part1.label.replace(-1, 0, inplace=True)
    data_part1 = data_part1.drop(['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis=1)
    x_part1 = data_part1.drop(['label'], axis=1)
    y_part1 = data_part1.label

    data_part2 = pd.read_csv('data/valid_data.csv')
    data_part2.drop_duplicates(inplace=True)
    data_part2.label.replace(-1, 0, inplace=True)
    data_part2 = data_part2.drop(['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis=1)
    x_part2 = data_part2.drop(['label'], axis=1)
    y_part2 = data_part2.label

    train_data = pd.concat([data_part1, data_part2],axis=0)
    x_train = train_data.drop(['label'], axis=1)
    y_train = train_data.label


    pred_data = pd.read_csv('data/test_data.csv')
    pred_data.drop_duplicates(inplace=True)
    output = pred_data[['User_id', 'Coupon_id', 'Date_received']]
    pred_data = pred_data.drop(['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis = 1)

    # print(x_train.columns)
    # print(pred_data.columns)
    data_part1 = xgb.DMatrix(x_part1, label=y_part1)
    data_part2 = xgb.DMatrix(x_part2, label=y_part2)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_pred = xgb.DMatrix(pred_data)

    start_time = time.time()
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

    # part1 for train and part2 for valid
    watchlist = [(data_part1,'train'),(data_part2,'val')]
    model = xgb.train(params,data_part1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)
    joblib.dump(model, "model/try_XGBoost_splittrain.m")

    # use both part1 and part2 for train
    # watchlist = [(data_train, 'train')]
    # model = xgb.train(params, data_train, num_boost_round=3500, evals=watchlist,early_stopping_rounds=300)
    # joblib.dump(model, "model/try_XGBoost_mergetrain.m")



    # 预测
    model = joblib.load("model/try_XGBoost_mergetrain.m")
    start_time = time.time()
    output['probability'] = model.predict(data_pred)
    output.probability = MinMaxScaler().fit_transform(output.probability)
    output.rename(columns={'User_id': 'user_id', 'Coupon_id': 'coupon_id', 'Date_received': 'date_received', },inplace=True)
    output.to_csv("result/try_xgb_mergetrain.csv", index=None)
    train_time = time.time() - start_time
    print('数据预测完成，耗时: %.3f秒' % train_time)

    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open('result/try_xgb_feature_score.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
    print('特征排序完成')
