# -*- coding: utf-8 -*-

# @Author: shu.wen
# @Date:   2016-05-30 15:31:15
# @Last Modified by:   shu.wen
# @Last Modified time: 2017-05-12 00:37:06
# 打分数据 /root/data/rule/RULE_USER_SKU_ACTION_SCORE_TBL.csv
__author__ = 'shu.wen'

import os
import sys
import csv
import time
import glob
import pandas
import math
import json
import numpy as np
# import pylab as pl
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_regression
import re
import pandas
from sklearn import tree
from sklearn import ensemble
import random
from sklearn.externals.six import StringIO
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from math import isnan
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
#import feature_trt_v5


if __name__ == '__main__':

    # step_01 读入数据集
    master_table_address = r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\RULE_USER_SKU_ACTION_SCORE_TBL.csv'
    master_df = pandas.read_csv(master_table_address, sep='|', header=0)


    print 'row_cnt: %d'%(master_df.shape[0])
    print
    
    # step_02 fill na

    master_df = master_df.fillna(-1)

    # step_02 load model
    lr = joblib.load(r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\lr.model')
    # step_03 dummy coding

    feature_set = [
        "USER_CATE_8_SKU_TYPE_5_CNT_15",
        #"USER_CATE_8_SKU_TYPE_5_RATE_15",
        "USER_CATE_8_SKU_SKU_CNT_3",
        "USER_CATE_8_SKU_HOUR_13_18_SECTION_3",
        "USER_CATE_8_SKU_WEEK_05_RATE_7",
        "USER_CATE_8_SKU_HOUR_13_18_CNT_3",
        "USER_CATE_8_SKU_HOUR_13_18_SECTION_CNT_7",
        #"USER_CATE_8_SKU_WEEK_03_CNT_15",
        #"USER_CATE_8_SKU_TYPE_6_RATE_7",
        #"USER_CATE_8_SKU_SKU_TYPE_1_RATE_3",

    ]

    master_df['Y_pre'] = lr.predict_proba(master_df[feature_set].values)[:,1]
    master_df_cut = master_df[['USER_ID', 'SKU_ID','Y_pre']].sort(columns =['USER_ID', 'Y_pre', 'SKU_ID', ], ascending=[1,0,1])
    
    master_df_cut.drop_duplicates(subset=['USER_ID'], keep = 'first', inplace = True )
    master_df_cut[['USER_ID', 'SKU_ID']].to_csv(r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\rule_submit.csv', sep=',', index=False )
    print 'done'
