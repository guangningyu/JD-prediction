#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
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

# ---------- Preprocessing ---------- #
def get_user():
    df = pd.read_csv(USER_DATA, sep=',', header=0)
    return df

def get_prod():
    df = pd.read_csv(PROD_DATA, sep=',', header=0)
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
        print df['user_reg_tm'].value_counts(dropna=False)

        sys.stdout = orig_stdout

def prof_prod():
    df = get_prod()
    print df


if __name__ == '__main__':
    prof_user()
    #prof_prod()

