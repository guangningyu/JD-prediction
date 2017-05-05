#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import numpy as np
import pandas as pd

# ---------- directories definition ---------- #
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def get_dir(dir_name):
    dir_path = os.path.join(MAIN_DIR, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

DATA_DIR = get_dir('data')
TEMP_DIR = get_dir('temp')
PROF_DIR = get_dir('prof')

# ---------- datasets definition ---------- #
USER_DATA = os.path.join(DATA_DIR, 'JData_User.csv')
PROD_DATA = os.path.join(DATA_DIR, 'JData_Product.csv')
COMMENT_DATA = os.path.join(DATA_DIR, 'JData_Comment.csv')
ACTION_DATA = os.path.join(DATA_DIR, 'JData_Action_*.csv')

MASTER_DATA = os.path.join(TEMP_DIR, 'master.csv')

# ---------- Preprocessing ---------- #
def get_user():
    df = pd.read_csv(USER_DATA, sep=',', header=0, encoding='GBK')
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'], errors='coerce')
    return df

def get_prod():
    df = pd.read_csv(PROD_DATA, sep=',', header=0, encoding='GBK')
    return df

def get_comment():
    df = pd.read_csv(COMMENT_DATA, sep=',', header=0, encoding='GBK')
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    return df

def get_action():
    files = glob.glob(ACTION_DATA)
    dfs = (pd.read_csv(file, sep=',', header=0, encoding='GBK') for file in files)
    df = pd.concat(dfs, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df[['user_id']] = df[['user_id']].astype(int)
    return df

# ---------- Profiling ---------- #
def prof_user():
    df = get_user()
    output_file = os.path.join(PROF_DIR, 'prof_user.txt')
    with open(output_file, 'wb') as f:
        orig_stdout = sys.stdout
        sys.stdout = f

        print '===== Check user data ====='

        print '\n> Check sample records...'
        print df.head(10)

        print '\n> Check column data type...'
        print df.dtypes

        print '\n> Count records...'
        print len(df)

        print '\n> Count unique user_id...'
        print len(df['user_id'].unique())

        print '\n> Count users by age...'
        print df['age'].value_counts(dropna=False)

        print '\n> Count users by sex...'
        print df['sex'].value_counts(dropna=False)

        print '\n> Count users by level...'
        print df['user_lv_cd'].value_counts(dropna=False)

        print '\n> Count users by reg date...'
        print df['user_reg_tm'].value_counts(dropna=False).sort_index()

        sys.stdout = orig_stdout

def prof_prod():
    df = get_prod()
    output_file = os.path.join(PROF_DIR, 'prof_product.txt')
    with open(output_file, 'wb') as f:
        orig_stdout = sys.stdout
        sys.stdout = f

        print '===== Check product data ====='

        print '\n> Check sample records...'
        print df.head(10)

        print '\n> Check column data type...'
        print df.dtypes

        print '\n> Count records...'
        print len(df)

        print '\n> Count unique sku_id...'
        print len(df['sku_id'].unique())

        print '\n> Count products by a1...'
        print df['a1'].value_counts(dropna=False)

        print '\n> Count products by a2...'
        print df['a2'].value_counts(dropna=False)

        print '\n> Count products by a3...'
        print df['a3'].value_counts(dropna=False)

        print '\n> Count products by category...'
        print df['cate'].value_counts(dropna=False)

        print '\n> Count products by brand...'
        print df['brand'].value_counts(dropna=False)

        sys.stdout = orig_stdout

def prof_comment():
    df = get_comment()
    output_file = os.path.join(PROF_DIR, 'prof_comment.txt')
    with open(output_file, 'wb') as f:
        orig_stdout = sys.stdout
        sys.stdout = f

        print '===== Check comment data ====='

        print '\n> Check sample records...'
        print df.head(10)

        print '\n> Check column data type...'
        print df.dtypes

        print '\n> Count records...'
        print len(df)

        print '\n> Count comments by dt...'
        print df['dt'].value_counts(dropna=False).sort_index()

        print '\n> Count unique sku_id...'
        print len(df['sku_id'].unique())

        print '\n> Count records by comment_num...'
        print df['comment_num'].value_counts(dropna=False)

        print '\n> Count records by has_bad_comment...'
        print df['has_bad_comment'].value_counts(dropna=False)

        print '\n> Count records by bad_comment_rate...'
        print df['bad_comment_rate'].value_counts(dropna=False).sort_index()

        sys.stdout = orig_stdout

def prof_action():
    df = get_action()
    output_file = os.path.join(PROF_DIR, 'prof_action.txt')
    with open(output_file, 'wb') as f:
        orig_stdout = sys.stdout
        sys.stdout = f

        print '===== Check action data ====='

        print '\n> Check sample records...'
        print df.head(10)

        print '\n> Check column data type...'
        print df.dtypes

        print '\n> Count records...'
        print len(df)

        print '\n> Count unique user_id...'
        print len(df['user_id'].unique())

        print '\n> Count unique sku_id...'
        print len(df['sku_id'].unique())

        print '\n> Count records by model_id...'
        print df['model_id'].value_counts(dropna=False)

        print '\n> Count records by type...'
        print df['type'].value_counts(dropna=False)

        print '\n> Count records by category...'
        print df['cate'].value_counts(dropna=False)

        print '\n> Count records by brand...'
        print df['brand'].value_counts(dropna=False)

        print '\n> Count records by time...'
        print df['time'].value_counts(dropna=False).sort_index()

        print '\n> Count unique sku_id (1.used to be ordered; 2.in cate8)...'
        print len(df[(df['type']==4) & (df['cate']==8)]['sku_id'].unique())

        print '\n> Count total orders (1.used to be ordered; 2.in cate8)...'
        print len(df[(df['type']==4) & (df['cate']==8)])

        print '\n> Count total orders by sku_id(1.used to be ordered; 2.in cate8)...'
        print df[(df['type']==4) & (df['cate']==8)]['sku_id'].value_counts(dropna=False)

        sys.stdout = orig_stdout

def get_session(outfile):
    print 'add session'
    # read action
    df = get_action()
    # get uniq sorted user_id * time pair
    df = df[['user_id', 'time']] \
         .drop_duplicates() \
         .sort_values(['user_id', 'time'], ascending=[True, True])
    # derive session_id
    session_num = 1
    def get_session_id(r):
        global session_num
        session_interval = 1800.0 # 30min
        time_diff = (r['time'] - r['last_time']) / np.timedelta64(1, 's')
        if r['user_id'] != r['last_user']:
            session_num = 1
        elif time_diff > session_interval:
            session_num += 1
        return session_num
    df['last_time'] = df['time'].shift(1)
    df['last_user'] = df['user_id'].shift(1)
    df['session_id'] = df.apply(lambda r : get_session_id(r), axis=1)
    df = df.drop(['last_time', 'last_user'], axis=1)
    # save to file
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_master(outfile):
    # read inputs
    user = get_user()
    prod = get_prod()
    comment = get_comment()
    action = get_action()

    # read session_id
    sess = pd.read_csv(MASTER_DATA + '_sess', sep=',', header=0, encoding='GBK')
    sess['time'] = pd.to_datetime(sess['time'], errors='coerce')

    # expand comments
    start_dt = '2016-02-01'
    end_dt = '2016-04-20'
    date_range = pd.DataFrame({'date': pd.date_range(start_dt, end_dt).format()})
    date_range['date'] = pd.to_datetime(date_range['date'], errors='coerce')
    date_range['week_start'] = date_range['date'].dt.to_period('W').apply(lambda r : r.start_time)
    comment = comment.merge(date_range, how='inner', left_on='dt', right_on='week_start')
    comment = comment.drop(['week_start', 'dt'], axis=1)

    # merge action, user, product and comment
    action['date'] = action['time'].dt.date
    action['date'] = pd.to_datetime(action['date'], errors='coerce')
    df = action.merge(user, how='left', on='user_id') \
                   .merge(prod, how='left', on='sku_id') \
                   .merge(comment, how='left', on=['date', 'sku_id']) \
                   .merge(sess, how='left', on=['user_id', 'time']) \
                   .rename(columns={
                       'cate_x':  'category',
                       'brand_x': 'brand',
                    }) \
                   .drop(['cate_y', 'brand_y'], axis=1) \
                   .sort_values(['user_id', 'time', 'sku_id', 'type', 'model_id'], ascending=[True, True, True, True, True])

    # save to file
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_train_input():
    # read master table
    #df = pd.read_csv(MASTER_DATA, sep=',', header=0, encoding='utf-8', nrows=30000) #TODO
    df = pd.read_csv(MASTER_DATA, sep=',', header=0, encoding='utf-8')

    # change column type
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'], errors='coerce')

    return df

if __name__ == '__main__':
    #prof_user()
    #prof_prod()
    #prof_comment()
    #prof_action()
    #get_session(MASTER_DATA + '_sess')
    get_master(MASTER_DATA)
    #get_train_input()

