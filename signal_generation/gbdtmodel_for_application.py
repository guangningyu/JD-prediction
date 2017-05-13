#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import GradientBoostingClassifier

reload(sys)
sys.setdefaultencoding('utf8')



# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    # model = GradientBoostingClassifier(n_estimators=200)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=2017)
    model.fit(train_x, train_y)
    return model



if __name__ == '__main__':
    print 'reading data...'

    x = pd.read_csv("/root/users/WSY/master_table_for_application.csv")

    num_row, num_feat = x.shape

    print '******************** Data Info *********************'
    print '# data: %d,  dimension: %d' % (num_row, num_feat)

    start_time = time.time()
    model = joblib.load('/root/users/WSY/trained_gbdt_model_0513_200trees_5')
    y = model.predict(x)
    print 'training took %fs!' % (time.time() - start_time)
    predictdf = pd.DataFrame(y, columns=['result'])
    outpath = '/root/users/WSY/predicted_for_application.csv'
    predictdf.to_csv(outpath, sep=",", index=False)

    outpath = '/root/users/WSY/expected_user_sku_pair_for_application.csv'
    result = x.drop(y[y['result'] == 0]].index)
    result.to_csv(outpath, sep=",", index=False)









