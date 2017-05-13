#!/usr/bin/env python

#from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import datetime
import pickle
import random
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ---------- path definition ---------- #
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(MAIN_DIR, 'temp')

# ---------- file definition ---------- #
MASTER_DATA = os.path.join(TEMP_DIR, 'master.csv')

MASTER_DATA_X = os.path.join(TEMP_DIR, 'master_x.csv')
MASTER_DATA_Y = os.path.join(TEMP_DIR, 'master_y.csv')
SKUS = os.path.join(TEMP_DIR, 'sku_list.csv')
USERS = os.path.join(TEMP_DIR, 'user_list.csv')
BRANDS = os.path.join(TEMP_DIR, 'brand_list.csv')
MODEL_IDS = os.path.join(TEMP_DIR, 'model_id_list.csv')

TRAIN_SEQUENCE = os.path.join(TEMP_DIR, 'train_sequence.pkl')
SCORE_SEQUENCE = os.path.join(TEMP_DIR, 'score_sequence.pkl')

TRAIN_LABELS = os.path.join(TEMP_DIR, 'train_labels.pkl')
SCORE_LABELS = os.path.join(TEMP_DIR, 'score_labels.pkl')

TRAINSET = os.path.join(TEMP_DIR, 'trainset.pkl')
TESTSET = os.path.join(TEMP_DIR, 'testset.pkl')
SCORESET = os.path.join(TEMP_DIR, 'scoreset.pkl')

TRAINSET_USER_RESULT = os.path.join(TEMP_DIR, 'trainset_user_result.pkl')
TESTSET_USER_RESULT = os.path.join(TEMP_DIR, 'testset_user_result.pkl')
SCORESET_USER_RESULT = os.path.join(TEMP_DIR, 'scoreset_user_result.pkl')

TRAINSET_SKU_RESULT = os.path.join(TEMP_DIR, 'trainset_sku_result.pkl')
TESTSET_SKU_RESULT = os.path.join(TEMP_DIR, 'testset_sku_result.pkl')
SCORESET_SKU_RESULT = os.path.join(TEMP_DIR, 'scoreset_sku_result.pkl')

TRAINSET_RESULT = os.path.join(TEMP_DIR, 'trainset_result.pkl')
TESTSET_RESULT = os.path.join(TEMP_DIR, 'testset_result.pkl')
SCORESET_RESULT = os.path.join(TEMP_DIR, 'scoreset_result.pkl')

SCORE_FILE = os.path.join(TEMP_DIR, 'score.csv')
OUTPUT_FILE = os.path.join(TEMP_DIR, 'upload.csv')

USER_STEP_RESULT = os.path.join(TEMP_DIR, 'user_step_result.pkl')
SKU_STEP_RESULT = os.path.join(TEMP_DIR, 'sku_step_result.pkl')

PROF_ACTION_NUM = os.path.join(TEMP_DIR, 'prof_action_num.csv')

# ---------- constants ---------- #
EVENT_LENGTH = 500
TOP_N_SKU = 100
TOP_N_BRAND = 20

# ---------- prepare training data ---------- #
def dump_pickle(dataset, save_file):
    with open(save_file, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(save_file):
    with open(save_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def load_csv(save_file):
    return pd.read_csv(save_file, sep=',', header=0, encoding='utf-8')

def separate_time_window(infile, outfile_x, outfile_y):
    # set time window
    start_dt = datetime.date(2016,2,1)
    cut_dt = datetime.date(2016,4,8)
    end_dt = datetime.date(2016,4,13)
    # read master
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'], errors='coerce')
    # separate by time window
    x = df[(df['date'] >= start_dt) & (df['date'] <= cut_dt)]
    y = df[(df['date'] > cut_dt) & (df['date'] <= end_dt)]
    x.to_csv(outfile_x, sep=',', index=False, encoding='utf-8')
    y.to_csv(outfile_y, sep=',', index=False, encoding='utf-8')

def get_skus(infile, outfile):
    # keep top TOP_N_SKU popular sku
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[(df['category']==8) & (df['type']==4)]
    df = df[['sku_id']] \
        .groupby('sku_id') \
        .size() \
        .to_frame(name = 'count') \
        .reset_index() \
        .sort_values(['count'], ascending=[False]) \
        .head(TOP_N_SKU) \
        .sort_values(['sku_id'], ascending=[True])
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_brands(infile, outfile):
    # keep top TOP_N_BRAND popular brands
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[['brand']] \
        .groupby('brand') \
        .size() \
        .to_frame(name = 'count') \
        .reset_index() \
        .sort_values(['count'], ascending=[False]) \
        .head(TOP_N_BRAND) \
        .sort_values(['brand'], ascending=[True])
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_model_ids(infile, outfile):
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[['model_id']] \
        .groupby('model_id') \
        .size() \
        .to_frame(name = 'count') \
        .reset_index() \
        .sort_values(['model_id'], ascending=[True])
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_users(infile, outfile):
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[['user_id']].drop_duplicates()
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_train_labels(user_file, sku_file, master_file, outfile):
    '''
    Result:
    [
        (202501, [1, 0], 168651, [0,0,0,...1,...0,0]),
        (202991, [0, 1],     -1, [0,0,0,...0,...0,0]),
        ...
    ]
    '''
    # 1.get all users who have order
    df = pd.read_csv(master_file, sep=',', header=0, encoding='utf-8')
    #   if a user has multiple orders, keep the latest one
    df = df[(df['category']==8) & (df['type']==4)] \
        .drop_duplicates(subset='user_id', keep='first')
    df = df[['user_id', 'sku_id']]
    df['has_order'] = 1
    # 2.append to user_list
    labels = pd.read_csv(user_file, sep=',', header=0, encoding='utf-8') \
        .merge(df, how='left', on='user_id')
    #   derive column1
    labels['is_positive'] = 0
    labels.loc[labels['has_order']>0, 'is_positive'] = 1
    #   derive column2
    labels['is_negative'] = 0
    labels.loc[pd.isnull(labels['has_order']), 'is_negative'] = 1
    # 3.add one hot encoding for sku list
    sku_df = pd.read_csv(sku_file, sep=',', header=0, encoding='utf-8')
    sku_list = sku_df['sku_id'].values.tolist()
    def get_sku_one_hot_encoding(sku_list, sku_id):
        encoding = [0] * len(sku_list)
        if sku_id in sku_list:
            encoding[sku_list.index(sku_id)] = 1
        return encoding
    # 4.convert to list
    user = labels['user_id'].values.tolist()
    label = labels[['is_positive', 'is_negative']].values.tolist()
    sku = [-1 if math.isnan(i) else int(i) for i in labels['sku_id'].values.tolist()]
    ordered_sku = [get_sku_one_hot_encoding(sku_list, sku_id) for sku_id in sku]
    labels = zip(user, label, sku, ordered_sku)
    # 5.dump data to pickle
    with open(outfile, 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def count_order_num_per_user(x_file, y_file, out_file):
    # count number of previous actions per user before target window
    df1 = pd.read_csv(x_file, sep=',', header=0, encoding='utf-8')
    df1 = df1[['user_id']] \
            .groupby('user_id') \
            .size() \
            .to_frame(name = 'count_action') \
            .reset_index() \
            .sort_values(['count_action'], ascending=[False])

    # count number of orders per user in target window
    df2 = pd.read_csv(y_file, sep=',', header=0, encoding='utf-8')
    df2 = df2[(df2['category']==8) & (df2['type']==4)]
    df2 = df2[['user_id']] \
            .groupby('user_id') \
            .size() \
            .to_frame(name = 'count_order') \
            .reset_index() \
            .sort_values(['count_order'], ascending=[False])

    # count number of previous actions (within 4 weeks) per user before target window
    start_dt = datetime.date(2016,3,12)
    df_temp = pd.read_csv(x_file, sep=',', header=0, encoding='utf-8')
    df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
    df3 = df_temp[(df_temp['date'] >= start_dt)]
    df3 = df3[['user_id']] \
            .groupby('user_id') \
            .size() \
            .to_frame(name = 'count_action_28') \
            .reset_index() \
            .sort_values(['count_action_28'], ascending=[False])

    # merge and save
    df = df1.merge(df2, how='left', on='user_id') \
            .merge(df3, how='left', on='user_id') \
            .sort_values(['count_order', 'count_action_28'], ascending=[False, True])
    df.to_csv(out_file, sep=',', index=False, encoding='utf-8')

def get_event_sequence(infile, outfile, keep_latest_events=200):
    '''
    Result:
    [
        (200002, array([[seq_500], [seq_499], ...,                  [seq_1]]), 500)
        (200003, array([[seq_36], ..., [seq_1], [fake_seq], ..., [fake_seq]]),  36)
        ...
    ]
    '''
    # 1.reverse the event history and keep latest events for each user
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df.sort_values(['user_id', 'time', 'sku_id', 'type', 'model_id'], ascending=[True, False, False, False, False]) \
        .groupby('user_id') \
        .head(keep_latest_events)
    #df.to_csv(MASTER_DATA + '_x_reverse', sep=',', index=False, encoding='utf-8')

    # 2.prepare sequence data
    def refactor_seq(seq, max_length):
        def padding(list):
            length = len(list)
            list += [0 for i in range(max_length - length)]
            return list
        s = []
        feature_num = len(seq[0])
        for i in range(feature_num):
            list = [action[i] for action in seq]
            list = padding(list)
            s += list
        return s

    # find the max datetime as observation timestamp
    max_timestamp = max(df['time'])
    max_timestamp = datetime.datetime.strptime(max_timestamp, '%Y-%m-%d %H:%M:%S')
    max_timestamp = int(max_timestamp.strftime('%s'))

    # init lists
    data = []
    user = []
    seq = []
    seq_len = []
    last_user_id = ''

    for index, row in df.iterrows():
        this_user_id = row['user_id']

        # preprocessing feature
        sku_id = row['sku_id']
        model_id = int(-1 if np.isnan(row['model_id']) else row['model_id'])
        type = row['type']
        category = row['category']
        brand = row['brand']
        a1 = int(0 if np.isnan(row['a1']) else row['a1'])
        a2 = int(0 if np.isnan(row['a2']) else row['a2'])
        a3 = int(0 if np.isnan(row['a3']) else row['a3'])
        timestamp = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
        timestamp = int(timestamp.strftime('%s'))
        till_obs = max_timestamp - timestamp
        if last_user_id != this_user_id: # for the very first record
            till_next = 9999999 # set it to a very large number, since there's no next action
        else:
            till_next = next_timestamp - timestamp
        next_timestamp = timestamp

        # create feature list
        action = [
            sku_id,
            model_id,
            type,
            category,
            brand,
            a1,
            a2,
            a3,
            till_next,
            till_obs,
        ]

        if last_user_id == '':
            user.append(this_user_id)
            seq.append(action)
        elif this_user_id == last_user_id:
            seq.append(action)
        else:
            # when meet new user
            user.append(this_user_id)
            seq_len.append(len(seq[:])) # append last user's seq_len
            data.append(refactor_seq(seq[:], keep_latest_events)) # append last user's seq
            seq = [] # init seq for the new user
            seq.append(action)
        last_user_id = this_user_id
    # append the last user
    seq_len.append(len(seq[:]))
    data.append(refactor_seq(seq[:], keep_latest_events))

    # 3.reshape and transpose
    size = len(data)
    n_steps = EVENT_LENGTH
    n_input = len(data[0]) / n_steps
    data = np.array(data).reshape(size, n_input, n_steps)
    data = np.transpose(data, (0,2,1)) # transpose of n_input and n_steps

    # 4.zip (user_id, data, seq_len) as data
    data = zip(user, data, seq_len)

    # 5.dump data to pickle
    with open(outfile, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 6.return sequence data
    return data

def split_train_test(data_pkl, labels_pkl, trainset, testset, train_rate=0.7):
    # load pickle
    data = load_pickle(data_pkl)
    labels = load_pickle(labels_pkl)
    # shuffle
    rows = zip(data, labels)
    random.shuffle(rows)
    # dump to pickle
    cut_point = int(train_rate * len(rows))
    dump_pickle(rows[:cut_point], trainset)
    dump_pickle(rows[cut_point:], testset)
    # print info
    print '> %s users in trainset' % len(rows[:cut_point])
    print '> %s users in testset'  % len(rows[cut_point:])
    print '> sample record:'
    print rows[:cut_point][0]

class SequenceData(object):
    """ Generate sequence of data with dynamic length.
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, dataset, sku_df, brand_df, label_type='order'):
        self.user   = [data[0] for (data, label) in dataset]
        self.data   = [data[1] for (data, label) in dataset]
        self.seqlen = [data[2] for (data, label) in dataset]

        self.order_label = [label[1] for (data, label) in dataset]
        self.sku_label   = [label[3] for (data, label) in dataset]

        self.sku_list   = sku_df['sku_id'].values.tolist()
        self.brand_list = brand_df['brand'].values.tolist()

        self.label_type = label_type
        self.batch_id = 0

        # get dataset stats
        u, d, s, l = self.next(1)
        self.length    = len(self.user)
        self.n_steps   = len(d[0])
        self.n_input   = len(d[0][0])
        self.n_classes = len(l[0])
        self.batch_id -= 1

    def transform_seq(self, seq):
        '''
        input shape: n_steps * n_input
        '''
        def norm_by_value(value, max_value):
            return [1.0 * value / max_value]

        def one_hot_encoding(value, value_list):
            default_list = [0] * (len(value_list) + 1)
            if value in value_list:
                default_list[value_list.index(value)] = 1
            else:
                # the last cell stands for other values
                default_list[-1] = 1
            return default_list

        def process_model_id(model_id):
            if model_id == -1:
                return -2
            elif model_id == 0:
                return -1
            else:
                return int(math.floor(1.0 * model_id / 100))

        seq_list = []
        for rec in seq:
            sku_id     = one_hot_encoding(int(rec[0]), self.sku_list)
            model_id   = one_hot_encoding(process_model_id(rec[1]), [-1, 0, 1, 2, 3])
            type       = one_hot_encoding(int(rec[2]), [1, 2, 3, 4, 5, 6])
            category   = one_hot_encoding(int(rec[3]), [4, 5, 6, 7, 8, 9])
            brand      = one_hot_encoding(int(rec[4]), self.brand_list)
            a1         = one_hot_encoding(int(rec[5]), [-1, 1, 2, 3])
            a2         = one_hot_encoding(int(rec[6]), [-1, 1, 2])
            a3         = one_hot_encoding(int(rec[7]), [-1, 1, 2])
            till_next  = norm_by_value(rec[8], 9999999)
            till_obs   = norm_by_value(rec[9], 9999999)
            # norm_rec
            norm_rec = sku_id + model_id + type + category + brand + a1 + a2 + a3 + till_next + till_obs
            seq_list.append(norm_rec)
        return np.array(seq_list)

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id + batch_size <= len(self.user):
            end_cursor = self.batch_id + batch_size
            batch_user   = self.user[self.batch_id:end_cursor]
            batch_data   = self.data[self.batch_id:end_cursor]
            batch_seqlen = self.seqlen[self.batch_id:end_cursor]
            if self.label_type == 'order':
                batch_label = self.order_label[self.batch_id:end_cursor]
            else:
                batch_label = self.sku_label[self.batch_id:end_cursor]
            self.batch_id += batch_size
        else:
            end_cursor = self.batch_id + batch_size - len(self.user)
            batch_user   = self.user[self.batch_id:] + self.user[:end_cursor]
            batch_data   = self.data[self.batch_id:] + self.data[:end_cursor]
            batch_seqlen = self.seqlen[self.batch_id:] + self.seqlen[:end_cursor]
            if self.label_type == 'order':
                batch_label = self.order_label[self.batch_id:] + self.order_label[:end_cursor]
            else:
                batch_label = self.sku_label[self.batch_id:] + self.sku_label[:end_cursor]
            self.batch_id = self.batch_id + batch_size - len(self.user)
        # do normalization & one-hot-encoding
        batch_data = [self.transform_seq(seq) for seq in batch_data]
        return batch_user, batch_data, batch_seqlen, batch_label

def run_rnn(trainset, testset, scoreset, trainset_result, testset_result, scoreset_result, step_file, training_iters=5000000, label_type='order'):
    # rnn parameters
    learning_rate = 0.01
    batch_size = 128
    display_step = 100
    n_hidden = 64 # hidden layer num of features

    # count input
    print '> %s records in trainset' % trainset.length
    print '> %s records in testset'  % testset.length
    print '> %s records in scoreset' % scoreset.length

    # model parameters
    n_steps   = trainset.n_steps
    n_input   = trainset.n_input
    n_classes = trainset.n_classes
    print 'n_steps:   %s' % n_steps
    print 'n_input:   %s' % n_input
    print 'n_classes: %s' % n_classes

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # define RNN model
    def dynamicRNN(x, seqlen, weights, biases):
        # prepare data shape to match `rnn` function requirements
        # current data input shape: (batch_size, n_steps, n_input)
        # required shape: `n_steps` tensors list of shape (batch_size, n_input)

        # unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

        # define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden)

        # get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

        # when performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.

        # `output` is a list of output at every timestep, we pack them in a tensor
        # and change back dimension to [batch_size, n_step, n_hidden]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * n_steps + (seqlen-1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    def RNN(x, seqlen, weights, biases):
        # prepare data shape to match `rnn` function requirements
        # current data input shape: (batch_size, n_steps, n_input)
        # required shape: `n_steps` tensors list of shape (batch_size, n_input)

        # unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

        # define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = dynamicRNN(x, seqlen, weights, biases)
    #pred = RNN(x, seqlen, weights, biases)

    # define results
    # why use softmax, not sigmoid: just one output unit to fire with a large value
    results = tf.nn.softmax(pred, name='results')

    # define loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # define optimizer (train step)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialzing the variables
    init = tf.global_variables_initializer()

    # set configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # launch the graph
    with tf.Session(config=config) as sess:
        sess.run(init)
        step = 1

        def cal_scores(dataset, batch_size):
            # create an empty list to contain output
            res = []
            # separate dataset to partitions, to avoid the out-of-memory issue
            rec_num = len(dataset.user)
            partition_num = int(math.ceil(1.0*rec_num/batch_size))
            # calculate results for each partition
            for i in range(partition_num):
                user, data, seqlength, label = dataset.next(batch_size)
                score = sess.run(results, feed_dict={x: data, y: label, seqlen: seqlength})
                part_res = zip(user, label, score)
                res += part_res
            # remove duplicated users
            uniq_res = []
            user_set = set([])
            for i in res:
                user_id = i[0]
                if user_id not in user_set:
                    uniq_res.append(i)
                user_set.add(user_id)
            return uniq_res

        print('> Start training...')
        # keep training until reach max iterations
        step_result = []
        while step * batch_size < training_iters:
            batch_user, batch_x, batch_seqlen, batch_y = trainset.next(batch_size)
            # run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            if step % display_step == 0:
                # calculate auc
                def cal_auc(score_list, label_type):
                    def get_sku_ind(rec):
                        sku_ind_list  = rec[1]
                        sku_prob_list = rec[2].tolist()
                        max_ind_index  = sku_ind_list.index(max(sku_ind_list))
                        max_prob_index = sku_prob_list.index(max(sku_prob_list))
                        if max_ind_index == max_prob_index:
                            return 1
                        else:
                            return 0

                    if label_type == 'order':
                        ind  = [i[1][0] for i in score_list]
                        prob = [i[2][0] for i in score_list]
                    else:
                        ind  = [get_sku_ind(i)     for i in score_list]
                        prob = [max(i[2].tolist()) for i in score_list]
                    fpr, tpr, thres = roc_curve(ind, prob, pos_label=1)
                    return auc(fpr, tpr)
                train_auc = cal_auc(cal_scores(trainset, batch_size), label_type=label_type)
                test_auc  = cal_auc(cal_scores(testset,  batch_size), label_type=label_type)

                # calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                # calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                print( \
                    "Iter %s, "                % str(step*batch_size) + \
                    "Minibatch Loss %.5f, "    % loss + \
                    "Training Accuracy %.5f, " % acc + \
                    "Training AUC %.5f, "      % train_auc + \
                    "Test AUC %.5f"            % test_auc \
                )
                # save step result
                step_result.append((step*batch_size, train_auc, test_auc))
            step += 1
        print("Optimization Finished!")
        # TODO calculate optimal iteration numbers using step_result

        # save result
        dump_pickle(step_result, step_file)
        dump_pickle(cal_scores(trainset, batch_size), trainset_result)
        dump_pickle(cal_scores(testset,  batch_size), testset_result)
        dump_pickle(cal_scores(scoreset, batch_size), scoreset_result)

def get_fake_labels(score_sequence, train_labels, save_file):
    fake_label = train_labels[0][1:]
    score_labels = [(i[0],) + fake_label for i in score_sequence]
    dump_pickle(score_labels, save_file)

def get_scoreset(score_sequence, score_labels, scoreset):
    rows = zip(load_pickle(score_sequence), load_pickle(score_labels))
    dump_pickle(rows, scoreset)
    print '> %s users in scoreset'  % len(rows)
    print '> sample record:'
    print rows[0]

def get_result(user_res, sku_res, sku_file, save_file):
    # format user level result
    user       = [i[0]    for i in user_res]
    order_ind  = [i[1][0] for i in user_res]
    order_prob = [i[2][0] for i in user_res]
    df1 = pd.DataFrame({
        'user_id'   : user,
        'order_ind' : order_ind,
        'order_prob': order_prob,
    })

    # format sku level result
    sku_df = pd.read_csv(sku_file, sep=',', header=0, encoding='utf-8')
    sku_list = sku_df['sku_id'].values.tolist()
    def get_sku_id(rec):
        if 1 in rec:
            return sku_list[rec.index(1)]
        else:
            return -1
    def guess_sku_id(rec):
        rec = rec.tolist()
        max_score = max(rec)
        sku_id = sku_list[rec.index(max_score)]
        return sku_id, max_score
    user         = [i[0]               for i in sku_res]
    sku_order_id = [get_sku_id(i[1])   for i in sku_res]
    sku_guess    = [guess_sku_id(i[2]) for i in sku_res]
    sku_guess_id    = [i[0] for i in sku_guess]
    sku_guess_score = [i[1] for i in sku_guess]
    df2 = pd.DataFrame({
        'user_id'        : user,
        'sku_order_id'   : sku_order_id,
        'sku_guess_id'   : sku_guess_id,
        'sku_guess_score': sku_guess_score,
    })
    def guess_right(row):
        if row['sku_order_id'] == row['sku_guess_id']:
            return 1
        else:
            return 0
    df2['guess_right'] = df2.apply(lambda row:guess_right(row), axis=1)

    # merge dfs
    result = df1.merge(df2, how='left', on='user_id') \
                .sort_values(['order_prob'], ascending=[False])
    dump_pickle(result, save_file)

def eval_roc(df):
    def plot_roc(prob, ind):
        # params
        lw = 2
        # calculate fpr, tpr, auc
        fpr, tpr, thres = roc_curve(ind, prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # plot roc curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (auc=%0.2f)' % roc_auc)
        plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    # check order prob
    prob = df['order_prob'].values.tolist()
    ind  = df['order_ind'].values.tolist()
    print '> Plot order prob ROC (%s records)...' % len(df)
    plot_roc(prob, ind)

    # check sku prob
    df2 = df[df['sku_order_id'] > 0] \
            .sort_values(['sku_guess_score'], ascending=[False])

    prob2 = df2['sku_guess_score'].values.tolist()
    ind2  = df2['guess_right'].values.tolist()
    print '> Plot sku prob ROC (%s records)...' % len(df2)
    plot_roc(prob2, ind2)

def gen_upload_result(trainset, testset, scoreset, save_file, score_file):
    def cal_precision_recall(dataset, cutoff):
        # select dataset according to cutoff
        dataset = dataset.sort_values(['order_prob'], ascending=[False])
        guessset = dataset[dataset['order_prob'] >= cutoff]
        # count values
        total_order       = sum(dataset['order_ind'].values.tolist())
        total_guess       = len(guessset)
        guess_order_right = max(1, sum(guessset['order_ind'].values.tolist()))
        guess_sku_right   = max(1, sum([1 for i in guessset['guess_right'].values.tolist() if i > 0.0]))
        # calculate precision and recall
        if total_guess > 0:
            f1_pre = 1.0 * guess_order_right / total_guess
            f1_rec = 1.0 * guess_order_right / total_order
            f2_pre = 1.0 * guess_sku_right   / total_guess
            f2_rec = 1.0 * guess_sku_right   / total_order
            # F1 value
            f1 = 6.0 * f1_rec * f1_pre / (5.0 * f1_rec + 1.0 * f1_pre)
            f2 = 5.0 * f2_rec * f2_pre / (2.0 * f2_rec + 3.0 * f2_pre)
            f  = 0.4 * f1 + 0.6 * f2
            return f1, f2, f
        else:
            return 0.0, 0.0, 0.0

    def select_cutoff(results):
        cutoff_list = [i[0] for i in results]
        f1_list     = [i[1] for i in results]
        f2_list     = [i[2] for i in results]
        f_list      = [i[3] for i in results]
        max_idx = f_list.index(max(f_list))
        optimal_cutoff = cutoff_list[max_idx]
        print '> The cutoff is %s, where:' % optimal_cutoff
        print '> f1 score: %s'             % f1_list[max_idx]
        print '> f2 score: %s'             % f2_list[max_idx]
        print '> f  score: %s'             % f_list[max_idx]
        return optimal_cutoff

    # calculate f score and select optimal cutoff
    interval = 0.0001
    loop_num = int(1.0 / interval)
    results = []
    for i in reversed(range(1, loop_num+1)):
        cutoff = 1.0 * i / loop_num
        f1, f2, f = cal_precision_recall(testset, cutoff)
        results.append((cutoff, f1, f2 ,f))
    optimal_cutoff = select_cutoff(results)

    # generate score file
    score_df = scoreset[['user_id', 'order_prob', 'sku_guess_id', 'sku_guess_score']] \
                .sort_values(['order_prob'], ascending=[False]) \
                .to_csv(score_file, sep=',', index=False, encoding='GBK')

    # generate upload file
    scoreset = scoreset.sort_values(['order_prob'], ascending=[False])
    scoreset = scoreset[scoreset['order_prob'] >= optimal_cutoff]
    scoreset = scoreset[['user_id', 'sku_guess_id']] \
                .rename(columns={'sku_guess_id': 'sku_id'})
    scoreset.to_csv(save_file, sep=',', index=False, encoding='GBK')

def eval_auc(res_list):
    df = pd.DataFrame({
        'train_auc': [i[1] for i in res_list],
        'test_auc':  [i[2] for i in res_list]
    }, index=[i[0] for i in res_list])
    df.plot(ylim=(0,1))
    df.head(30).plot(ylim=(0,1))
    plt.show()


if __name__ == '__main__':

    # ---------- Data Preparation ---------- #

    # 1.split time window
    #separate_time_window(MASTER_DATA, MASTER_DATA_X, MASTER_DATA_Y) # 20min
    #get_users(MASTER_DATA_X, USERS) # 2min
    #get_skus(MASTER_DATA_Y, SKUS) # 3min
    #get_brands(MASTER_DATA_Y, BRANDS) # 3min

    # 2.prepare input sequence
    #get_event_sequence(MASTER_DATA_X, TRAIN_SEQUENCE, keep_latest_events=EVENT_LENGTH) # 83min
    #get_event_sequence(MASTER_DATA,   SCORE_SEQUENCE, keep_latest_events=EVENT_LENGTH) # 89min

    # 3.prepare labels
    #get_train_labels(USERS, SKUS, MASTER_DATA_Y, TRAIN_LABELS) # 2min
    #get_fake_labels(load_pickle(SCORE_SEQUENCE), load_pickle(TRAIN_LABELS), SCORE_LABELS) # 1min

    # 4.merge input sequence & labels
    #split_train_test(TRAIN_SEQUENCE, TRAIN_LABELS, TRAINSET, TESTSET, 0.5) # 2.5min
    #get_scoreset(SCORE_SEQUENCE, SCORE_LABELS, SCORESET) # 0.1min

    # ---------- Model Training ---------- #

    # 1.train, test & score at user level
    trainset = load_pickle(TRAINSET)
    testset  = load_pickle(TESTSET)
    scoreset = load_pickle(SCORESET)
    # create objects
    trainset = SequenceData(trainset, load_csv(SKUS), load_csv(BRANDS), label_type='order')
    testset  = SequenceData(testset,  load_csv(SKUS), load_csv(BRANDS), label_type='order')
    scoreset = SequenceData(scoreset, load_csv(SKUS), load_csv(BRANDS), label_type='order')
    run_rnn(trainset, testset, scoreset, TRAINSET_USER_RESULT, TESTSET_USER_RESULT, SCORESET_USER_RESULT, USER_STEP_RESULT, training_iters=3000000, label_type='order') # 590min for 5000000 iters

    # 2.train, test & score at sku level
    ## select users who have orders and the ordered sku_id is in sku list
    #trainset = [i for i in load_pickle(TRAINSET) if sum(i[1][3]) > 0]
    #testset  = [i for i in load_pickle(TESTSET)  if sum(i[1][3]) > 0]
    #scoreset = [i for i in load_pickle(SCORESET)]
    ## create objects
    #trainset = SequenceData(trainset, load_csv(SKUS), load_csv(BRANDS), label_type='sku')
    #testset  = SequenceData(testset,  load_csv(SKUS), load_csv(BRANDS), label_type='sku')
    #scoreset = SequenceData(scoreset, load_csv(SKUS), load_csv(BRANDS), label_type='sku')
    #run_rnn(trainset, testset, scoreset, TRAINSET_SKU_RESULT, TESTSET_SKU_RESULT, SCORESET_SKU_RESULT, SKU_STEP_RESULT, training_iters=210000, label_type='sku') # 217min for 5000000 iters

    # ---------- Model Evaluation ---------- #

    # 1.Merge user level & sku level result
    #get_result(load_pickle(TRAINSET_USER_RESULT), load_pickle(TRAINSET_SKU_RESULT), SKUS, TRAINSET_RESULT)
    #get_result(load_pickle(TESTSET_USER_RESULT), load_pickle(TESTSET_SKU_RESULT), SKUS, TESTSET_RESULT)
    #get_result(load_pickle(SCORESET_USER_RESULT), load_pickle(SCORESET_SKU_RESULT), SKUS, SCORESET_RESULT)

    # 2.Check train & test auc for each step
    #eval_auc(load_pickle(USER_STEP_RESULT))
    #eval_auc(load_pickle(SKU_STEP_RESULT))

    # 3.Check final roc curve
    #eval_roc(load_pickle(TRAINSET_RESULT))
    #eval_roc(load_pickle(TESTSET_RESULT))

    # 4.Select cutoff and generate upload file
    #gen_upload_result(load_pickle(TRAINSET_RESULT), load_pickle(TESTSET_RESULT), load_pickle(SCORESET_RESULT), OUTPUT_FILE, SCORE_FILE)

    # ---------- No Longer Needed ---------- #
    #count_order_num_per_user(MASTER_DATA_X, MASTER_DATA_Y, PROF_ACTION_NUM) # 5min
    #get_model_ids(MASTER_DATA_Y, MODEL_IDS) # 3min

