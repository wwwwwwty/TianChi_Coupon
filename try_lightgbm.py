import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':

    data_part1 = pd.read_csv('data/train_data.csv')
    data_part1.drop_duplicates(inplace=True)
    data_part1.label.replace(-1, 0, inplace=True)
    data_part1 = data_part1.drop(
        ['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis=1)
    x_part1 = data_part1.drop(['label'], axis=1)
    y_part1 = data_part1.label

    data_part2 = pd.read_csv('data/valid_data.csv')
    data_part2.drop_duplicates(inplace=True)
    data_part2.label.replace(-1, 0, inplace=True)
    data_part2 = data_part2.drop(
        ['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis=1)
    x_part2 = data_part2.drop(['label'], axis=1)
    y_part2 = data_part2.label

    train_data = pd.concat([data_part1, data_part2], axis=0)
    x_train = train_data.drop(['label'], axis=1)
    y_train = train_data.label

    pred_data = pd.read_csv('data/test_data.csv')
    pred_data.drop_duplicates(inplace=True)
    output = pred_data[['User_id', 'Coupon_id', 'Date_received']]
    pred_data = pred_data.drop(
        ['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'day_gap_before', 'day_gap_after'], axis=1)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_part1, y_part1)
    lgb_eval = lgb.Dataset(x_part2, y_part2, reference=lgb_train)
    lgb_train = lgb.Dataset(x_train, y_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'is_unbalance': 'true',
        'metric': {'l2', 'auc'},
        'min_data_in_leaf': 10,
        'num_leaves': 31, # num_leaves = 2 ^ depth
        'num_iterations': 500,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'feature_fraction_seed': 13,
        'bagging_fraction': 1,
        'bagging_freq': 10,
        'bagging_seed': 13,
        'verbose': 0,
        'num_boost_round': 2000,
        'early_stopping_rounds': 50
    }

    gbm = lgb.train(params, lgb_train, valid_sets = lgb_eval)
    gbm.save_model('model/try_lightgbm_splittrain.m')

    # gbm = lgb.train(params, lgb_train)
    # gbm.save_model('model/try_lightgbm_mergetrain.m')

    # dump model to json (and save to file)
    model_json = gbm.dump_model()
    with open('model/lightmodel.json', 'w+') as f:
        json.dump(model_json, f, indent=4)
    print('Feature importances:', list(gbm.feature_importance()))

    # # predict
    output['probability'] = gbm.predict(pred_data, num_iteration=gbm.best_iteration)
    output.probability = MinMaxScaler().fit_transform(output.probability)
    # print(output['probability'])
    print(len(output[output.probability > 0.5]))
    print(len(output[output.probability < 0.5]))
    output.rename(columns={'User_id': 'user_id', 'Coupon_id': 'coupon_id', 'Date_received': 'date_received', },
                  inplace=True)
    output.to_csv("result/try_lightgbm_splittrain.csv", index=None)
    # output.to_csv("result/try_lightgbm_mergetrain.csv", index=None)
    # # eval
    # print('The rmse of prediction is:', mean_squared_error(y_part2, y_pred) ** 0.5)

