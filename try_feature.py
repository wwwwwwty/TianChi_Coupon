import pandas as pd
import numpy as np
from datetime import date
import time

# 不提示copy的warning
pd.options.mode.chained_assignment = None

def get_other_feature(dataset3):

    #### User related feature ####
    t = dataset3[['User_id']]
    # 用户收到优惠券数量
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    # print(t)

    t1 = dataset3[['User_id', 'Coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()
    # print(t1)

    t2 = dataset3[['User_id', 'Coupon_id', 'Date_received']]
    t2['Date_received_all'] = t2.Date_received.astype('str')
    # 用:连接日期
    t2 = t2.groupby(['User_id', 'Coupon_id'])['Date_received_all'].agg(
        lambda x: ':'.join(x) if (str(x).find("null") == -1) else ' ').reset_index()
    t2['receive_number'] = t2.Date_received_all.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.Date_received_all.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.Date_received_all.apply(lambda s: min([int(d) for d in s.split(':')]))
    # t2 = t2[['User_id', 'Coupon_id', 'Date_received_all', 'max_date_received', 'min_date_received']]
    t2 = t2[['User_id', 'Coupon_id', 'max_date_received', 'min_date_received']]

    def calc_date_gap(from_day, to_day):
        from_day = str(from_day)
        to_day = str(to_day)
        # skip index
        from_day = from_day.split()
        from_day = from_day[1]
        to_day = to_day.split()
        to_day = to_day[1]
        gap = (date(int(from_day[0:4]), int(from_day[4:6]), int(from_day[6:8])) - date(int(to_day[0:4]), int(to_day[4:6]), int(to_day[6:8]))).days
        print(from_day, to_day, gap)
        return gap

    t3 = dataset3[dataset3.Date_received != 'null'][['User_id','Coupon_id','Date_received']]
    t3 = pd.merge(t3, t2, on=['User_id', 'Coupon_id'], how='left')
    # t3 = t3[t3.Date_received_all.notnull()]
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.Date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone'] = t3.Date_received.astype('int') - t3.min_date_received

    def is_firstlastone(x):
        if x == 0:
            return 1
        elif x > 0:
            return 0
        else:
            return -1  # those only receive once

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
    t3 = t3[['User_id', 'Coupon_id', 'Date_received', 'this_month_user_receive_same_coupon_lastone', 'this_month_user_receive_same_coupon_firstone']]
    # print(t3)

    # 收到的优惠券数
    t4 = dataset3[['User_id', 'Date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['User_id', 'Date_received']).agg('sum').reset_index()

    # 一天收到的优惠券数
    t5 = dataset3[['User_id', 'Coupon_id', 'Date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['User_id', 'Coupon_id', 'Date_received']).agg('sum').reset_index()

    t6 = dataset3[['User_id', 'Coupon_id', 'Date_received']]
    t6.date_received = t6.Date_received.astype('str')
    t6 = t6.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'Date_received': 'dates'}, inplace=True)

    def get_day_gap_before(s):
        date_received, dates = s.split('-')
        dates = dates.split(':')
        gaps = []
        for d in dates:
            this_gap = (
            date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]),
                                                                                                   int(d[4:6]),
                                                                                                   int(d[6:8]))).days
            if this_gap > 0:
                gaps.append(this_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    def get_day_gap_after(s):
        date_received, dates = s.split('-')
        dates = dates.split(':')
        gaps = []
        for d in dates:
            this_gap = (
            date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]),
                                                               int(date_received[6:8]))).days
            if this_gap > 0:
                gaps.append(this_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    t7 = dataset3[['User_id', 'Coupon_id', 'Date_received']]
    t7 = pd.merge(t7, t6, on=['User_id', 'Coupon_id'], how='left')
    t7 = t7[t7.Date_received != 'null']  # diff from
    t7['date_received_date'] = t7.Date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['User_id', 'Coupon_id', 'Date_received', 'day_gap_before', 'day_gap_after']]
    # print(t7)

    other_feature3 = pd.merge(t1, t, on='User_id')
    other_feature3 = pd.merge(other_feature3, t3, on=['User_id', 'Coupon_id'])
    other_feature3 = pd.merge(other_feature3, t4, on=['User_id', 'Date_received'])
    other_feature3 = pd.merge(other_feature3, t5, on=['User_id', 'Coupon_id', 'Date_received'])
    other_feature3 = pd.merge(other_feature3, t7, on=['User_id', 'Coupon_id', 'Date_received'])
    # print(other_feature3)
    return other_feature3

def get_coupon_feature(dataset3):

    #### Coupon related feature ####
    def calc_discount_rate(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return float(s[0])
        else:
            return 1.0 - float(s[1]) / float(s[0])

    def get_discount_man(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 'null'
        else:
            return int(s[0])

    def get_discount_jian(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 'null'
        else:
            return int(s[1])

    def is_man_jian(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 0
        else:
            return 1

    coupon3_feature = dataset3[dataset3['Date_received'] != 'null']
    # 第几周
    coupon3_feature['day_of_week'] = coupon3_feature.Date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    # 第几个月
    coupon3_feature['day_of_month'] = coupon3_feature.Date_received.astype('str').apply(lambda x: int(x[6:8]))
    # 发券日期距离结束的时间
    coupon3_feature['days_distance'] = coupon3_feature.Date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days)
    coupon3_feature['discount_man'] = coupon3_feature.Discount_rate.apply(get_discount_man)
    coupon3_feature['discount_jian'] = coupon3_feature.Discount_rate.apply(get_discount_jian)
    coupon3_feature['is_man_jian'] = coupon3_feature.Discount_rate.apply(is_man_jian)
    coupon3_feature['discount_rate'] = coupon3_feature.Discount_rate.apply(calc_discount_rate)

    d = coupon3_feature[['Coupon_id']]
    d['Coupon_count'] = 1
    d = d.groupby('Coupon_id').agg('sum').reset_index()
    coupon3_feature = pd.merge(coupon3_feature, d, on='Coupon_id', how='left')
    # print(coupon3_feature)
    return coupon3_feature

def get_merchant_feature(data):

    #### merchant related features ####
    merchant3 = data[['Merchant_id', 'Coupon_id', 'Distance', 'Date_received', 'Date']]
    t = merchant3[['Merchant_id']]
    t.drop_duplicates(inplace=True)
    # print(t)

    # 消费的次数
    t1 = merchant3[merchant3.Date != 'null'][['Merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('Merchant_id').agg('sum').reset_index()
    # print(t1)

    # 发出并使用的优惠券
    t2 = merchant3[(merchant3.Date != 'null') & (merchant3.Coupon_id != 'null')][['Merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('Merchant_id').agg('sum').reset_index()
    # print(t2)

    # 发出的优惠券
    t3 = merchant3[merchant3.Coupon_id != 'null'][['Merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('Merchant_id').agg('sum').reset_index()
    # print(t3)

    # 被使用优惠券的商家距离
    t4 = merchant3[(merchant3.Date != 'null') & (merchant3.Coupon_id != 'null')][['Merchant_id', 'Distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.Distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    # print(t4)

    # 商家最近距离 距离不是一样吗？
    t5 = t4.groupby('Merchant_id').agg('min').reset_index()
    t5.rename(columns={'Distance': 'merchant_min_distance'}, inplace=True)
    # print(t5)

    t6 = t4.groupby('Merchant_id').agg('max').reset_index()
    t6.rename(columns={'Distance': 'merchant_max_distance'}, inplace=True)

    # t7 = t4.groupby('Merchant_id').agg('mean').reset_index()
    # t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    # t8 = t4.groupby('Merchant_id').agg('median').reset_index()
    # t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant3_feature = pd.merge(t, t1, on='Merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t2, on='Merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t3, on='Merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t5, on='Merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t6, on='Merchant_id', how='left')
    # merchant3_feature = pd.merge(merchant3_feature, t7, on='Merchant_id', how='left')
    # merchant3_feature = pd.merge(merchant3_feature, t8, on='Merchant_id', how='left')

    merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    # 使用的优惠券占发出的优惠券比例
    merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype(
        'float') / merchant3_feature.total_coupon
    # 使用优惠券消费次数占总消费次数的比例
    merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype(
        'float') / merchant3_feature.total_sales
    merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan, 0)
    # print(merchant3_feature)
    return merchant3_feature

def get_user_feature(data):

    #### User related feature ####
    def get_user_date_datereceived_gap(s):
        s = s.split(':')
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                            int(s[1][6:8]))).days

    user3 = data[['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received', 'Date']]
    # print(user3)

    t = user3[['User_id']]
    t.drop_duplicates(inplace=True)
    # print(t)

    # 用户消费数
    t1 = user3[user3.Date != 'null'][['User_id', 'Merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.Merchant_id = 1
    t1 = t1.groupby('User_id').agg('sum').reset_index()
    t1.rename(columns={'Merchant_id': 'Count_merchant'}, inplace=True)
    # print(t1)

    # 使用优惠券用户与商家的距离
    t2 = user3[(user3.Date != 'null') & (user3.Coupon_id != 'null')][['User_id', 'Distance']]
    t2.replace('null', -1, inplace=True)
    t2.Distance = t2.Distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    # print(t2)

    # 最近距离
    t3 = t2.groupby('User_id').agg('min').reset_index()
    t3.rename(columns={'Distance': 'user_min_distance'}, inplace=True)
    # print(t3)

    # 最远距离
    t4 = t2.groupby('User_id').agg('max').reset_index()
    t4.rename(columns={'Distance': 'user_max_distance'}, inplace=True)
    # print(t4)

    # 平均距离
    t5 = t2.groupby('User_id').agg('mean').reset_index()
    t5.rename(columns={'Distance': 'user_mean_distance'}, inplace=True)
    # print(t5)

    # 中位数距离
    t6 = t2.groupby('User_id').agg('median').reset_index()
    t6.rename(columns={'Distance': 'user_median_distance'}, inplace=True)

    # 使用优惠券数量
    t7 = user3[(user3.Date != 'null') & (user3.Coupon_id != 'null')][['User_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('User_id').agg('sum').reset_index()
    # print(t7)

    # 购买总数
    t8 = user3[user3.Date != 'null'][['User_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('User_id').agg('sum').reset_index()

    # 收到优惠券数
    t9 = user3[user3.Coupon_id != 'null'][['User_id']]
    t9['Coupon_received'] = 1
    t9 = t9.groupby('User_id').agg('sum').reset_index()

    # 使用优惠券的时间间隔
    t10 = user3[(user3.Date_received != 'null') & (user3.Date != 'null')][['User_id', 'Date_received', 'Date']]
    t10['user_date_datereceived_gap'] = t10.Date + ':' + t10.Date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['User_id', 'user_date_datereceived_gap']]
    # print(t10)

    # 平均使用间隔
    t11 = t10.groupby('User_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    # 最小间隔
    t12 = t10.groupby('User_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    # 最大间隔
    t13 = t10.groupby('User_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user3_feature = pd.merge(t, t1, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t3, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t4, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t5, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t6, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t7, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t8, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t9, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t11, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t12, on='User_id', how='left')
    user3_feature = pd.merge(user3_feature, t13, on='User_id', how='left')

    user3_feature.Count_merchant = user3_feature.Count_merchant.replace(np.nan, 0)
    user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan, 0)
    user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype(
        'float') / user3_feature.buy_total.astype('float')
    user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype(
        'float') / user3_feature.Coupon_received.astype('float')
    user3_feature.buy_total = user3_feature.buy_total.replace(np.nan, 0)
    user3_feature.Coupon_received = user3_feature.Coupon_received.replace(np.nan, 0)
    # print(user3_feature)
    return user3_feature

def get_user_merchant_feature(data):

    #### User Merchant related feature ####
    all_user_merchant = data[['User_id', 'Merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    user_merchant = data

    # 消费总次数
    t = user_merchant[['User_id', 'Merchant_id', 'Date']]
    t = t[t.Date != 'null'][['User_id', 'Merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)
    # print(t)

    # 收到指定的优惠券数
    t1 = user_merchant[['User_id', 'Merchant_id', 'Coupon_id']]
    t1 = t1[t1.Coupon_id != 'null'][['User_id', 'Merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)
    # print(t1)

    # 使用制定优惠券数
    t2 = user_merchant[['User_id', 'Merchant_id', 'Date', 'Date_received']]
    t2 = t2[(t2.Date != 'null') & (t2.Date_received != 'null')][['User_id', 'Merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = user_merchant[['User_id', 'Merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)
    # print(t3)

    # 正常消费没用优惠券数
    t4 = user_merchant[['User_id', 'Merchant_id', 'Date', 'Coupon_id']]
    t4 = t4[(t4.Date != 'null') & (t4.Coupon_id == 'null')][['User_id', 'Merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)
    # print(t4)

    user_merchant3 = pd.merge(all_user_merchant, t, on=['User_id', 'Merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t1, on=['User_id', 'Merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t2, on=['User_id', 'Merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t3, on=['User_id', 'Merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t4, on=['User_id', 'Merchant_id'], how='left')

    user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_received.astype('float')
    user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')
    user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype(
        'float') / user_merchant3.user_merchant_any.astype('float')
    user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')

    # print(user_merchant3)
    return user_merchant3

def merge_and_padding(other_feature, user_feature, coupon_feature, merchant_feature, user_merchant_feature, is_train = True):

    all_features = pd.merge(coupon_feature, merchant_feature, on='Merchant_id', how='left')
    # print(len(all_features))
    all_features = pd.merge(all_features, user_feature, on='User_id', how='left')
    # print(len(all_features))
    all_features = pd.merge(all_features, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    # print(len(all_features))
    all_features = pd.merge(all_features, other_feature, on=['User_id', 'Coupon_id', 'Date_received'], how='left')
    # print(len(all_features))
    # all_features.drop_duplicates(inplace=True)
    # print(len(all_features))

    all_features.user_merchant_buy_total = all_features.user_merchant_buy_total.replace(np.nan, 0)
    all_features.user_merchant_any = all_features.user_merchant_any.replace(np.nan, 0)
    all_features.user_merchant_received = all_features.user_merchant_received.replace(np.nan, 0)

    all_features['is_weekend'] = all_features.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(all_features.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    all_features = pd.concat([all_features, weekday_dummies], axis=1)
    print(len(all_features))

    def get_label(s):
        s = s.split(':')
        # 没有用优惠券
        if s[0] == 'null':
            return 0
        # 15天内使用优惠券
        elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                          int(s[1][6:8]))).days <= 15:
            return 1
        # 超过15天使用优惠券
        else:
            return -1

    # 设置标记值
    if is_train == True:
        all_features['label'] = all_features.Date.astype('str') + ':' + all_features.Date_received.astype('str')
        all_features.label = all_features.label.apply(get_label)
        # print(all_features['label'])

    if is_train == True:
        all_features.drop(['Merchant_id', 'day_of_week', 'Date', 'Coupon_count'], axis=1,
                          inplace=True)
    else:
        all_features.drop(['Merchant_id', 'day_of_week', 'Coupon_count'], axis=1,
                          inplace=True)
    all_features = all_features.replace('null', np.nan)
    # print(len(all_features))

    return all_features



if __name__ == "__main__":

    offline_train_file = 'data/ccf_offline_stage1_train.csv'
    test_file = 'data/ccf_offline_stage1_test_revised.csv'
    pd.set_option('display.width', 200)

    train = pd.read_csv(offline_train_file)
    # 有领券
    train_data = train[(train.Date_received >= '20160414') & (train.Date_received <= '20160514')]
    # 消费+领券但没消费
    train_data_feature = train[(train.Date >= '20160101') & (train.Date <= '20160413') | ((train.Date == 'null') & (train.Date_received >= '20160101') & (train.Date_received <= '20160413'))]

    valid_data = train[(train.Date_received >= '20160515') & (train.Date_received <= '20160615')]
    valid_data_feature = train[(train.Date >= '20160201') & (train.Date <= '20160514') | ((train.Date == 'null') & (train.Date_received >= '20160201') & (valid_data.Date_received <= '20160514'))]

    test_data = pd.read_csv(test_file)
    test_data_feature = train[((train.Date>='20160315')&(train.Date<='20160630'))|((train.Date=='null')&(train.Date_received>='20160315')&(train.Date_received<='20160630'))]
    test_data['Date_received'] = test_data['Date_received'].astype('str')
    # data = data.loc[0:1000, :]

    # 清洗训练数据
    start_time = time.time()
    train_other_feature = get_other_feature(train_data)
    train_coupon_feature = get_coupon_feature(train_data)
    train_user_feature = get_user_feature(train_data_feature)
    train_merchant_feature = get_merchant_feature(train_data_feature)
    train_user_merchant_feature = get_user_merchant_feature(train_data_feature)

    train_final_data = merge_and_padding(train_other_feature,
                                         train_user_feature,
                                         train_coupon_feature,
                                         train_merchant_feature,
                                         train_user_merchant_feature)
    # print(final_data)
    time_usage = time.time() - start_time
    print('训练数据清洗完成，耗时%f秒' % time_usage)
    train_final_data.to_csv('data/train_data.csv', index=None)

    # 清洗验证数据
    start_time = time.time()
    valid_other_feature = get_other_feature(valid_data)
    valid_coupon_feature = get_coupon_feature(valid_data)
    valid_user_feature = get_user_feature(valid_data_feature)
    valid_merchant_feature = get_merchant_feature(valid_data_feature)
    valid_user_merchant_feature = get_user_merchant_feature(valid_data_feature)

    valid_final_data = merge_and_padding(valid_other_feature,
                                         valid_user_feature,
                                         valid_coupon_feature,
                                         valid_merchant_feature,
                                         valid_user_merchant_feature)
    # print(final_data)
    time_usage = time.time() - start_time
    print('验证数据清洗完成，耗时%f秒' % time_usage)
    valid_final_data.to_csv('data/valid_data.csv', index=None)


    # 清洗测试数据
    start_time = time.time()
    test_other_feature = get_other_feature(test_data)
    # print(len(test_other_feature))
    test_coupon_feature = get_coupon_feature(test_data)
    # print(len(test_coupon_feature))
    test_user_feature = get_user_feature(test_data_feature)
    # print(len(test_user_feature))
    test_merchant_feature = get_merchant_feature(test_data_feature)
    # print(len(test_merchant_feature))
    test_user_merchant_feature = get_user_merchant_feature(test_data_feature)
    # print(len(test_user_merchant_feature))

    test_final_data = merge_and_padding(test_other_feature,
                                        test_user_feature,
                                        test_coupon_feature,
                                        test_merchant_feature,
                                        test_user_merchant_feature,
                                        is_train = False)
    time_usage = time.time() - start_time
    print('测试数据清洗完成，耗时%f秒' % time_usage)
    test_final_data.to_csv('data/test_data.csv', index=None)


