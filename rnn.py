#!/usr/bin/env python

#from __future__ import print_function

import os
#import tensorflow as tf
#import random
import numpy as np
import pandas as pd
import datetime
import pickle

# ---------- table definition ---------- #
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(MAIN_DIR, 'temp')
MASTER_DATA = os.path.join(TEMP_DIR, 'master.csv')
MASTER_DATA_X = os.path.join(TEMP_DIR, 'master_x.csv')
MASTER_DATA_Y = os.path.join(TEMP_DIR, 'master_y.csv')
SKUS = os.path.join(TEMP_DIR, 'sku_list.csv')
USERS = os.path.join(TEMP_DIR, 'user_list.csv')
USER_LABELS = os.path.join(TEMP_DIR, 'user_labels.pkl')
EVENT_SEQUENCE = os.path.join(TEMP_DIR, 'event_sequence.pkl')

# ---------- constants ---------- #
EVENT_LENGTH = 300

# ---------- prepare training data ---------- #
def separate_time_window(infile, outfile_x, outfile_y):
    # set time window
    start_dt = datetime.date(2016,2,1)
    cut_dt = datetime.date(2016,4,10)
    end_dt = datetime.date(2016,4,15)
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
        .sort_values(['count'], ascending=[False])
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_users(infile, outfile):
    df = pd.read_csv(infile, sep=',', header=0, encoding='utf-8')
    df = df[['user_id']].drop_duplicates()
    df.to_csv(outfile, sep=',', index=False, encoding='utf-8')

def get_user_labels(user, master, outfile):
    # 1.get all users who have order
    df = pd.read_csv(master, sep=',', header=0, encoding='utf-8')
    #   if a user has multiple orders, keep the latest one
    df = df[(df['category']==8) & (df['type']==4)] \
        .drop_duplicates(subset='user_id', keep='last')
    df = df[['user_id']]
    df['has_order'] = 1
    # 2.append to user_list
    labels = pd.read_csv(user, sep=',', header=0, encoding='utf-8') \
        .merge(df, how='left', on='user_id')
    #   derive column1
    labels['is_positive'] = 0
    labels.loc[labels['has_order']>0, 'is_positive'] = 1
    #   derive column2
    labels['is_negative'] = 0
    labels.loc[pd.isnull(labels['has_order']), 'is_negative'] = 1
    # 3.convert to list
    labels = labels[['is_positive', 'is_negative']].values.tolist()
    # 4.dump data to pickle
    with open(outfile, 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return labels

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

def get_user_event_sequence(infile, outfile, keep_latest_events=200):
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

    data = []
    user = []
    seq = []
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
        action = [
            sku_id,
            model_id,
            type,
            category,
            brand,
            a1,
            a2,
            a3,
        ]

        if last_user_id == '':
            user.append(this_user_id)
            seq.append(action)
        elif this_user_id == last_user_id:
            seq.append(action)
        else:
            user.append(this_user_id)
            data.append(refactor_seq(seq[:], keep_latest_events))
            seq = []
            seq.append(action)
        last_user_id = this_user_id
    # append the last user
    data.append(refactor_seq(seq[:], keep_latest_events))

    # 3.dump data to pickle
    with open(outfile, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 4.return sequence data
    return data

class SequenceData(object):
    """ Generate sequence of data with dynamic length.
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, data_pkl, labels_pkl):
        # read pickles
        with open(data_pkl, 'rb') as handle:
            self.data = pickle.load(handle)
        with open(labels_pkl, 'rb') as handle:
            self.labels = pickle.load(handle)
        #self.seqlen = []
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id + batch_size <= len(self.data):
            batch_data = self.data[self.batch_id:(self.batch_id + batch_size)]
            batch_labels = self.labels[self.batch_id:(self.batch_id + batch_size)]
            self.batch_id += batch_size
        else:
            batch_data = self.data[self.batch_id:] + self.data[:(self.batch_id + batch_size - len(self.data))]
            batch_labels = self.labels[self.batch_id:] + self.labels[:(self.batch_id + batch_size - len(self.data))]
            self.batch_id = self.batch_id + batch_size - len(self.data)
        return batch_data, batch_labels


if __name__ == '__main__':
    #separate_time_window(MASTER_DATA, MASTER_DATA_X, MASTER_DATA_Y) # 20min
    #get_skus(MASTER_DATA, SKUS) # 3min
    #get_users(MASTER_DATA_X, USERS) # 2min
    #labels = get_user_labels(USERS, MASTER_DATA_Y, USER_LABELS) # 0.1min
    #data = get_user_event_sequence(MASTER_DATA_X, EVENT_SEQUENCE, keep_latest_events=EVENT_LENGTH) # 51min
    trainset = SequenceData(EVENT_SEQUENCE, USER_LABELS)

    # ---------- no longer needed ---------- #
    #count_order_num_per_user(MASTER_DATA_Y) # 0.1min


# ==========
#   MODEL
# ==========

## Parameters
#learning_rate = 0.01
#training_iters = 1000000
#batch_size = 128
#display_step = 10
#
## Network Parameters
#seq_max_len = 20 # Sequence max length
#n_hidden = 64 # hidden layer num of features
#n_classes = 2 # linear sequence or not
#
#trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
#print(trainset.data)
#
#
## tf Graph input
#x = tf.placeholder("float", [None, seq_max_len, 1])
#y = tf.placeholder("float", [None, n_classes])
## A placeholder for indicating each sequence length
#seqlen = tf.placeholder(tf.int32, [None])
#
## Define weights
#weights = {
#    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#}
#biases = {
#    'out': tf.Variable(tf.random_normal([n_classes]))
#}
#
#
#def dynamicRNN(x, seqlen, weights, biases):
#
#    # Prepare data shape to match `rnn` function requirements
#    # Current data input shape: (batch_size, n_steps, n_input)
#    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
#
#    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#    x = tf.unstack(x, seq_max_len, 1)
#
#    # Define a lstm cell with tensorflow
#    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
#
#    # Get lstm cell output, providing 'sequence_length' will perform dynamic
#    # calculation.
#    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
#                                sequence_length=seqlen)
#
#    # When performing dynamic calculation, we must retrieve the last
#    # dynamically computed output, i.e., if a sequence length is 10, we need
#    # to retrieve the 10th output.
#    # However TensorFlow doesn't support advanced indexing yet, so we build
#    # a custom op that for each sample in batch size, get its length and
#    # get the corresponding relevant output.
#
#    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
#    # and change back dimension to [batch_size, n_step, n_input]
#    outputs
#    outputs = tf.stack(outputs)
#    outputs = tf.transpose(outputs, [1, 0, 2])
#
#    # Hack to build the indexing and retrieve the right output.
#    batch_size = tf.shape(outputs)[0]
#    # Start indices for each sample
#    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
#    # Indexing
#    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
#
#    # Linear activation, using outputs computed above
#    return tf.matmul(outputs, weights['out']) + biases['out']
#
#pred = dynamicRNN(x, seqlen, weights, biases)
#
## Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#
## Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
## Initializing the variables
#init = tf.global_variables_initializer()
#
## Launch the graph
#with tf.Session() as sess:
#    sess.run(init)
#    step = 1
#    # Keep training until reach max iterations
#    while step * batch_size < training_iters:
#        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
#        # Run optimization op (backprop)
#        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                       seqlen: batch_seqlen})
#        if step % display_step == 0:
#            # Calculate batch accuracy
#            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
#                                                seqlen: batch_seqlen})
#            # Calculate batch loss
#            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
#                                             seqlen: batch_seqlen})
#            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))
#        step += 1
#    print("Optimization Finished!")
#
#    # Calculate accuracy
#    test_data = testset.data
#    test_label = testset.labels
#    test_seqlen = testset.seqlen
#    print("Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
#                                      seqlen: test_seqlen}))

