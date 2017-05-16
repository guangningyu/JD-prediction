#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

mst = pd.read_csv('/root/data/USER_LIST_0416_0420.csv', sep=",")
sku_master = pd.read_csv('/root/data/SKU_MASTER_0201_0415.csv', sep=",")

mst = mst.merge(sku_master, how='left', on=['SKU_ID'])
out_file = '/root/data/MODEL_MASTER_0201_0415_step1.csv'
mst.to_csv(out_file, sep=",", index=False)

del sku_master

user_action = pd.read_csv('/root/data/USER_ACTION_0201_0415.csv', sep=",")

mst = pd.merge(mst, user_action, how='left', on=['USER_ID'])
out_file = '/root/data/MODEL_MASTER_0201_0415_step2.csv'
mst.to_csv(out_file, sep=",", index=False)

del user_action

user_sku_action = pd.read_csv('/root/data/SKU_ACTION_0201_0410.csv', sep=",")

mst = pd.merge(mst, user_sku_action, how='left', on=['USER_ID', 'SKU_ID'])
out_file = '/root/data/MODEL_MASTER_0201_0415.csv'
mst.to_csv(out_file, sep=",", index=False)

del user_sku_action
