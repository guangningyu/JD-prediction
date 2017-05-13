import pandas as pd
import numpy as np
from datetime import *

# MODEL DATA
readin_file1 = '/root/data/LABEL_0409_0413.csv'
rdin_file1 = pd.read_csv(readin_file1, sep=",")
# 141125 records, 1008 Y=1, 140117 Y=0

user_list = rdin_file1['USER_ID'].drop_duplicates()

readin_file2 = '/root/data/SKU_MASTER_0201_0408.csv'
rdin_file2 = pd.read_csv(readin_file2, sep=",")
# 27653 records

readin_file3 = '/root/data/USER_ACTION_0201_0408.csv'
rdin_file3 = pd.read_csv(readin_file3, sep=",")

#readin_file4 = '/root/data/USER_SKU_ACTION_0201_0408.csv'
#rdin_file4 = pd.read_csv(readin_file4, sep=",")

print(len(rdin_file1))
# 141125
mst = rdin_file1.merge(rdin_file2, how='left', on='SKU_ID')
print(len(mst))
# 141125

mst2 = mst.merge(rdin_file3, how='left', on='USER_ID')
print(len(mst))
# 141125

out_file = '/root/data/MODEL_INPUT_MASTER_0201_0408.csv'
mst2.to_csv(out_file, sep=",", index=False)

in_1 = '/root/data/MODEL_INPUT_MASTER_0201_0408.csv'
in_2 = '/root/data/USER_SKU_ACTION_0201_0408.csv'
df = pd.read_csv(in_2, sep=",")

df_2 = df.merge(user_list,how='inner', on='USER_ID')

df_1 = pd.read_csv(in_1, sep=",")
df_2 = pd.read_csv(in_2, sep=",")

print(len(df_1))
# 141125
print(len(df_2))
# 1147219
mst = df_1.merge(df_2, how='left', on=['USER_ID', 'SKU_ID'])
print(len(mst))
# 141125


out_file = '/root/data/MODEL_MASTER_0201_0408.csv'
mst.to_csv(out_file, sep=",", index=False)



# SCORING DATA
readin_file1 = '/root/data/LABEL_0409_0413.csv'
rdin_file1 = pd.read_csv(readin_file1, sep=",")
# 141125 records, 1008 Y=1, 140117 Y=0

user_list = rdin_file1['USER_ID'].drop_duplicates()

readin_file2 = '/root/data/SKU_MASTER_0201_0408.csv'
rdin_file2 = pd.read_csv(readin_file2, sep=",")
# 27653 records

readin_file3 = '/root/data/USER_ACTION_0201_0408.csv'
rdin_file3 = pd.read_csv(readin_file3, sep=",")

#readin_file4 = '/root/data/USER_SKU_ACTION_0201_0408.csv'
#rdin_file4 = pd.read_csv(readin_file4, sep=",")

print(len(rdin_file1))
# 141125
mst = rdin_file1.merge(rdin_file2, how='left', on='SKU_ID')
print(len(mst))
# 141125

mst2 = mst.merge(rdin_file3, how='left', on='USER_ID')
print(len(mst))
# 141125

out_file = '/root/data/MODEL_INPUT_MASTER_0201_0408.csv'
mst2.to_csv(out_file, sep=",", index=False)

in_1 = '/root/data/MODEL_INPUT_MASTER_0201_0408.csv'
in_2 = '/root/data/USER_SKU_ACTION_0201_0408.csv'
df = pd.read_csv(in_2, sep=",")

df_2 = df.merge(user_list,how='inner', on='USER_ID')

df_1 = pd.read_csv(in_1, sep=",")
df_2 = pd.read_csv(in_2, sep=",")

print(len(df_1))
# 141125
print(len(df_2))
# 1147219
mst = df_1.merge(df_2, how='left', on=['USER_ID', 'SKU_ID'])
print(len(mst))
# 141125

out_file = '/root/data/MODEL_MASTER_0201_0408.csv'
mst.to_csv(out_file, sep=",", index=False)
