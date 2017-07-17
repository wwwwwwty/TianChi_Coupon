import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import csv
import time
import sklearn.metrics as metrics
from sklearn.utils import check_random_state
from scipy.sparse import hstack,vstack
from sklearn import svm
from sklearn.externals import joblib



def fit_sample(X, y):
    """Resample the dataset.
    """
    label = np.unique(y)
    stats_c_ = {}
    maj_n = 0
    maj_c_= 0
    for i in label:
        nk = sum(y == i)
        stats_c_[i] = nk
        if nk > maj_n:
            maj_n = nk
            maj_c_ = i

    # Keep the samples from the majority class
    X_resampled = X[y == maj_c_]
    y_resampled = y[y == maj_c_]
    # Loop over the other classes over picking at random
    for key in stats_c_.keys():

        # If this is the majority class, skip it
        if key == maj_c_:
            continue

        # Define the number of sample to create
        num_samples = int(stats_c_[maj_c_] - stats_c_[key])

        # Pick some elements at random
        random_state = check_random_state(42)
        indx = random_state.randint(low=0, high=stats_c_[key], size=num_samples)

        # Concatenate to the majority class
        X_resampled = vstack([X_resampled, X[y == key], X[y == key][indx]])
        np.shape(y_resampled), np.shape(y[y == key]), np.shape(y[y == key][indx])
        y_resampled = np.array(list(y_resampled) + list(y[y == key]) + list(y[y == key][indx]))
    return X_resampled, y_resampled

def print_result(train_type, y_test, y_hat, train_time):

    acc = accuracy_score(y_test, y_hat)
    pre = precision_score(y_test, y_hat)
    rec = recall_score(y_test, y_hat)
    fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_hat.ravel())
    auc = metrics.auc(fpr, tpr)
    print('模型为: ',train_type)
    print('准确率：%.4f' % acc)
    print('精确率：%.4f' % pre)
    print('召回率：%.4f' % rec)
    print('AUC：%.4f' % auc)
    print('耗时: %.3f秒' % train_time)

if __name__ == "__main__":

    offline_train_file = 'data/New_offline_train.csv'
    pd.set_option('display.width', 200)

    data = pd.read_csv(offline_train_file)
    # with pd.option_context('display.float_format', lambda x: '%.1f' % x):
    #     print(data.describe())
    data = data.loc[:50000, :]
    x = data[['Discount_value1', 'Discount_value2', 'Distance']]
    y = data['Consume15']
    x = np.array(x)
    y = np.array(y)

    # 重采样作为训练集
    # x_train, y_train = fit_sample(x, y)
    # 在原数据集中选择一部分作为测试集
    # _, x_test, _, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # train_type = ''
    # train_type = 'Logistic'
    # train_type = 'RandomForest'
    # train_type = 'XGBoost'
    train_type = 'SVM'

    # 线性回归
    if  train_type == 'Logistic':
        start_time = time.time()
        lr = LogisticRegression(penalty='l2')
        lr.fit(x_train, y_train)
        y_hat = lr.predict(x_test)
        train_time = time.time() - start_time
        print_result(train_type, y_hat, y_test, train_time)
        joblib.dump(lr, "model/Logistic.m")

    # 随机森林
    if train_type == 'RandomForest':
        start_time = time.time()
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(x_train, y_train)
        y_hat = rfc.predict(x_test)
        train_time = time.time() - start_time
        print_result(train_type, y_hat, y_test, train_time)
        joblib.dump(rfc, "model/RandomForest.m")

    # XGboost
    if train_type == 'XGBoost':
        start_time = time.time()
        data_train = xgb.DMatrix(x_train, label=y_train)
        data_test = xgb.DMatrix(x_test, label=y_test)
        watch_list = [(data_test, 'eval'), (data_train, 'train')]
        # alpha:l1正则惩罚系数 lambda:l2正则惩罚系数
        param = {'max_depth': 8, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
        # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
        bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
        y_hat = bst.predict(data_test)
        y_hat[y_hat > 0.5] = 1
        y_hat[~(y_hat > 0.5)] = 0
        train_time = time.time() - start_time
        print_result(train_type, y_hat, y_test, train_time)
        joblib.dump(bst, "model/XGBoost.m")

    # SVM
    if train_type == 'SVM':
        start_time = time.time()
        svmclf = svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={0: 1, 1: 15}, probability=True)
        svmclf.fit(x_train, y_train)
        y_hat = svmclf.predict(x_test)
        train_time = time.time() - start_time
        print_result(train_type, y_hat, y_test, train_time)
        joblib.dump(svmclf, "model/SVM.m")

