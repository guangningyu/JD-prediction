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
TRAIN_SEQUENCE = os.path.join(TEMP_DIR, 'train_sequence.pkl')
TRAIN_LABELS = os.path.join(TEMP_DIR, 'train_labels.pkl')
SCORE_SEQUENCE = os.path.join(TEMP_DIR, 'score_sequence.pkl')
SCORE_LABELS = os.path.join(TEMP_DIR, 'score_labels.pkl')
TRAINSET = os.path.join(TEMP_DIR, 'trainset.pkl')
TESTSET = os.path.join(TEMP_DIR, 'testset.pkl')
SCORESET = os.path.join(TEMP_DIR, 'scoreset.pkl')
TESTSET_USER_RESULT = os.path.join(TEMP_DIR, 'testset_user_result.pkl')
SCORESET_USER_RESULT = os.path.join(TEMP_DIR, 'scoreset_user_result.pkl')
TESTSET_SKU_RESULT = os.path.join(TEMP_DIR, 'testset_sku_result.pkl')
SCORESET_SKU_RESULT = os.path.join(TEMP_DIR, 'scoreset_sku_result.pkl')
TESTSET_RESULT = os.path.join(TEMP_DIR, 'testset_result.pkl')
SCORESET_RESULT = os.path.join(TEMP_DIR, 'scoreset_result.pkl')

# ---------- constants ---------- #
EVENT_LENGTH = 500

# ---------- prepare training data ---------- #
def dump_pickle(dataset, save_file):
    with open(save_file, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(save_file):
    with open(save_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

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
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[(df['category']==8) & (df['type']==4)]
    df = df[['sku_id']] \
        .groupby('sku_id') \
        .size() \
        .to_frame(name = 'count') \
        .reset_index() \
        .sort_values(['sku_id'], ascending=[True])
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

def count_order_num_per_user(infile):
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[(df['category']==8) & (df['type']==4)]
    df = df[['user_id']] \
        .groupby('user_id') \
        .size() \
        .to_frame(name = 'count') \
        .reset_index() \
        .sort_values(['count'], ascending=[False])
    print df

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
            list += [-999999 for i in range(max_length - length)]
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
            till_next = 99999999 # set it to a very large number, since there's no next action
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

class SequenceData(object):
    """ Generate sequence of data with dynamic length.
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, dataset, label_type='order'):
        self.user   = [data[0] for (data, label) in dataset]
        self.data   = [data[1] for (data, label) in dataset]
        self.seqlen = [data[2] for (data, label) in dataset]

        self.order_label = [label[1] for (data, label) in dataset]
        self.sku_label   = [label[3] for (data, label) in dataset]

        self.label_type = label_type
        self.batch_id = 0

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
        return batch_user, batch_data, batch_seqlen, batch_label

def run_rnn(trainset, testset, scoreset, testset_result, scoreset_result, label_type='order'):
    # rnn parameters
    learning_rate = 0.01
    training_iters = 1000000
    batch_size = 500
    display_step = 10
    n_hidden = 64 # hidden layer num of features

    # model parameters
    n_steps = len(trainset.data[0])
    n_input = len(trainset.data[0][0])
    if label_type == 'order':
        n_classes = len(trainset.order_label[0])
    else:
        n_classes = len(trainset.sku_label[0])

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
        # and change back dimension to [batch_size, n_step, n_input]
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

    #pred = dynamicRNN(x, seqlen, weights, biases) #TODO
    pred = RNN(x, seqlen, weights, biases)

    # define results
    # why use softmax, not sigmoid: just one output unit to fire with a large value
    results = tf.nn.softmax(pred, name='results')

    # define loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # define optimizer (train step)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialzing the variables
    init = tf.global_variables_initializer()

    # launch the graph
    with tf.Session() as sess:
        print('> Start training...')
        sess.run(init)
        step = 1
        # keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_user, batch_x, batch_seqlen, batch_y = trainset.next(batch_size)
            # run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            if step % display_step == 0:
                # calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                # calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # save result
        def save_result(dataset, save_file):
            user = dataset.user
            data = dataset.data
            seqlength = dataset.seqlen
            if label_type == 'order':
                label = dataset.order_label
            else:
                label = dataset.sku_label
            score = sess.run(results, feed_dict={x: data, y: label, seqlen: seqlength})
            res = zip(user, label, score)
            dump_pickle(res, save_file)
        save_result(testset, testset_result)
        save_result(scoreset, scoreset_result)

def get_fake_labels(score_sequence, train_labels, save_file):
    fake_label = train_labels[0][1:]
    score_labels = [(i[0],) + fake_label for i in score_sequence]
    dump_pickle(score_labels, save_file)

def get_scoreset(score_sequence, score_labels, scoreset):
    rows = zip(load_pickle(score_sequence), load_pickle(score_labels))
    dump_pickle(rows, scoreset)
    print '> %s users in scoreset'  % len(rows)

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

def eval_result(df):
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
    plot_roc(prob, ind)

    # check sku prob
    df2 = df[df['sku_order_id'] != -1] \
            .sort_values(['sku_guess_score'], ascending=[False])
    prob = df2['sku_guess_score'].values.tolist()
    ind  = df2['guess_right'].values.tolist()
    plot_roc(prob, ind)


if __name__ == '__main__':
    # ---------- prepare train+test sequence & labels ---------- #
    #separate_time_window(MASTER_DATA, MASTER_DATA_X, MASTER_DATA_Y) # 20min
    #get_users(MASTER_DATA_X, USERS) # 2min
    #get_skus(MASTER_DATA, SKUS) # 3min
    #get_event_sequence(MASTER_DATA_X, TRAIN_SEQUENCE, keep_latest_events=EVENT_LENGTH) # 83min
    #get_train_labels(USERS, SKUS, MASTER_DATA_Y, TRAIN_LABELS) # 2min

    # ---------- prepare scoring sequence & labels ---------- #
    #get_event_sequence(MASTER_DATA, SCORE_SEQUENCE, keep_latest_events=EVENT_LENGTH) # 89min
    #get_fake_labels(load_pickle(SCORE_SEQUENCE), load_pickle(TRAIN_LABELS), SCORE_LABELS) # 1min

    # ---------- prepare trainset, testset & scoreset ---------- #
    #split_train_test(TRAIN_SEQUENCE, TRAIN_LABELS, TRAINSET, TESTSET, 0.5) # 2.5min
    #get_scoreset(SCORE_SEQUENCE, SCORE_LABELS, SCORESET) # 0.1min

    # ---------- train, test & score at user level ---------- #
    #trainset = SequenceData(load_pickle(TRAINSET), label_type='order')
    #testset  = SequenceData(load_pickle(TESTSET),  label_type='order')
    #scoreset = SequenceData(load_pickle(SCORESET), label_type='order')
    #run_rnn(trainset, testset, scoreset, TESTSET_USER_RESULT, SCORESET_USER_RESULT, label_type='order') # 5min

    # ---------- train, test & score at sku level ---------- #
    ## select users who have orders for trainset
    #trainset = [i for i in load_pickle(TRAINSET) if i[1][2] != -1]
    #testset  = [i for i in load_pickle(TESTSET)]
    #scoreset = [i for i in load_pickle(SCORESET)]
    ## create objects
    #trainset = SequenceData(trainset, label_type='sku')
    #testset  = SequenceData(testset,  label_type='sku')
    #scoreset = SequenceData(scoreset, label_type='sku')
    #run_rnn(trainset, testset, scoreset, TESTSET_SKU_RESULT, SCORESET_SKU_RESULT, label_type='sku') # 7min

    # ---------- evaluation ---------- #
    #get_result(load_pickle(TESTSET_USER_RESULT), load_pickle(TESTSET_SKU_RESULT), SKUS, TESTSET_RESULT)
    #get_result(load_pickle(SCORESET_USER_RESULT), load_pickle(SCORESET_SKU_RESULT), SKUS, SCORESET_RESULT)
    eval_result(load_pickle(TESTSET_RESULT))

    # ---------- no longer needed ---------- #
    #count_order_num_per_user(MASTER_DATA_Y) # 0.1min

