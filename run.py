#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
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

# ---------- Preprocessing ---------- #
def get_user():
    df = pd.read_csv(USER_DATA, sep=',', header=0)
    return df

def get_prod():
    df = pd.read_csv(PROD_DATA, sep=',', header=0)
    return df

def get_comment():
    df = pd.read_csv(COMMENT_DATA, sep=',', header=0)
    return df

def get_action():
    files = glob.glob(ACTION_DATA)
    dfs = (pd.read_csv(file, sep=',', header=0) for file in files)
    df = pd.concat(dfs, ignore_index=True)
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

        sys.stdout = orig_stdout


if __name__ == '__main__':
    prof_user()
    prof_prod()
    prof_comment()
    prof_action()

