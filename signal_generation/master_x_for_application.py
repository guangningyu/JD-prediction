#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import numpy as np

# raw_file = pd.read_csv('/root/users/WSY/master_table_v1.csv')
raw_file = pd.read_csv('/root/data/MODEL_MASTER_0201_0415.csv')

num_rows, num_cols =  raw_file.shape
print 'rows: %d' % num_rows
print 'cols: %d' % num_cols


def age_divide(string_num):
    if string_num == '-1':
        return -1
    if string_num == '15岁以下':
        return 1
    if string_num == '16-25岁':
        return 2
    if string_num == '26-35岁':
        return 3
    if string_num == '36-45岁':
        return 4
    if string_num == '46-55岁':
        return 5
    if string_num == '56岁以上':
        return 6

raw_file['AGE_USABLE'] = raw_file['AGE'].apply(age_divide)

raw_file = raw_file.drop('AGE', 1)
raw_file = raw_file.drop('USER_ID', 1)
raw_file = raw_file.drop('SKU_ID', 1)

raw_file = raw_file.astype(np.float64, copy=False)
raw_file = raw_file.replace([np.inf, -np.inf], np.nan).fillna(0)
raw_file = (raw_file - raw_file.mean())/raw_file.std()
raw_file = raw_file.replace([np.inf, -np.inf], np.nan).fillna(0)

out_file = '/root/users/WSY/master_table_for_application.csv'
raw_file.to_csv(out_file, sep=",", index=False, header=False)

