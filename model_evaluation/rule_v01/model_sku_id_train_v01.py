# -*- coding: utf-8 -*-

# @Author: shu.wen
# @Date:   2016-05-30 15:31:15
# @Last Modified by:   shu.wen
# @Last Modified time: 2017-05-12 00:37:06
# 训练数据 /root/data/rule/RULE_USER_SKU_ACTION_MST_TBL_v2.csv
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
    master_table_address = r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\RULE_USER_SKU_ACTION_MST_TBL_v2.csv'
    master_df = pandas.read_csv(master_table_address, sep=',', header=0)


    print 'row_cnt: %d, trt_rt: %.4f'%(master_df.shape[0], master_df.Y.sum()*1.0/master_df.shape[0])
    print
    
    # step_02 fill na

    master_df['SKU_A1'] = master_df['SKU_A1'].apply(int).apply(str)
    master_df['SKU_A2'] = master_df['SKU_A2'].apply(int).apply(str)
    master_df['SKU_A3'] = master_df['SKU_A3'].apply(int).apply(str)
    master_df = master_df.fillna(-1)


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

    vec = DictVectorizer(sparse = False)
    master_df_vec = vec.fit_transform(master_df[feature_set].to_dict(orient = 'record'))


    # vec.get_feature_names()

    # step_02 划分数据集

    X_train, X_test, y_train, y_test = train_test_split(master_df_vec, master_df['Y'], test_size=0.5, random_state=int(time.time()))

    # scaler = StandardScaler().fit(X_train_raw)
    # X_train = scaler.transform(X_train_raw)
    # X_test = scaler.transform(X_test_raw)


    print u'训练集行数：%d, target rate: %f'%(X_train.shape[0], sum(y_train)*1.0/X_train.shape[0])
    print u'测试集行数：%d, target rate: %f'%(X_test.shape[0], sum(y_test)*1.0/X_test.shape[0])

    # step_02 model

    #clf = GradientBoostingClassifier(loss='exponential', learning_rate = 0.001, n_estimators = 2, subsample = 1.0, criterion='friedman_mse', max_depth= 3, max_features="log2",  presort='auto')
    clf = LogisticRegression(penalty = 'l1', tol = 0.001, max_iter = 1000, C = 1.0, class_weight = {1:30, 0:1},)
    
    clf = clf.fit(X_train, y_train)
    joblib.dump(clf, r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\lr.model')

    
    #feature_importances = zip(vec.get_feature_names(), clf.feature_importances_.tolist())
    #feature_importances = [ item for item in feature_importances]
    #feature_importances.sort(key=lambda x:x[1], reverse = True)    
    #feature_importances_df = pandas.DataFrame(feature_importances)
    #feature_importances_df.columns = ['feature', 'importances']
    #feature_importances_df.to_csv(r'C:\shu.wen\opera\00_projects\99_competition\01_JData\data\rule\feature_importances_part_2.csv', sep=',', index=False )    
    
    
    print
    y_train_pre = clf.predict_proba(X_train)
    fpr_train, tpr_train, thres_train = roc_curve( y_train, y_train_pre[:,1])

    y_test_pre = clf.predict_proba(X_test)
    fpr_test, tpr_test, thres_test = roc_curve( y_test, y_test_pre[:,1])

    print auc(fpr_train, tpr_train)
    print auc(fpr_test, tpr_test)
    print
    
    
    
    
    for item in range(10):
        thres = item*1.0/10
        P1_1_train = precision_score(y_train, y_train_pre[:,1]>=thres )
        R1_1_train = recall_score(y_train, y_train_pre[:,1]>=thres)
        F11_train = 5*R1_1_train*P1_1_train/(2*R1_1_train+3*P1_1_train)
        P1_1_test = precision_score(y_test, y_test_pre[:,1]>=thres )
        R1_1_test = recall_score(y_test, y_test_pre[:,1]>=thres)
        F11_test = 5*R1_1_test*P1_1_test/(2*R1_1_test+3*P1_1_test)

        print "Thres %s"%(thres)
        print 'Train F11: %.4f, P: %.4f, R: %.4f'%(F11_train, P1_1_train, R1_1_train)
        print 'Test F11: %.4f, P: %.4f, R: %.4f'%(F11_test, P1_1_test, R1_1_test)
        print
    
