# -*- coding: utf-8 -*-
# @Time    : 2018/9/25 19:11
# @Author  : Jason
# @FileName: predict.py

from __future__ import print_function

import os
import sys

import tensorflow as tf
import tensorflow.contrib.keras as kr

from Text.cnews_helper import read_categories, read_vocab
from Text.cnn_model import CNNConfig, CNN
from Text.rnn_model import RNNConfig, RNN

#
# try:
#     bool(type(unicode))
# except NameError:
#     unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
predict_dir = os.path.join(base_dir, 'cnews.predict.txt')

save_dir = 'checkpoint'


# save_path = os.path.join(save_dir, 'best_validation')


class Model:
    def __init__(self, model_name):
        if model_name == 'cnn':
            self.config = CNNConfig()
        else:
            self.config = RNNConfig()
        self.categories, self.cat_to_id = read_categories()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        if model_name == 'cnn':
            self.model = CNN(self.config)
        else:
            self.model = RNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=os.path.join(save_dir, model_name, 'best_validation'))

    def predict(self, message):
        content = str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.dropout_keep_prob: 1.0
        }
        y_pred_class = self.session.run(self.model.y_predict_class, feed_dict=feed_dict)
        return self.categories[y_pred_class[0]]


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['cnn', 'rnn']:
        raise ValueError("""Usage: python predict.py cnn/rnn""")
    if sys.argv[1] == 'cnn':
        model = Model('cnn')
    elif sys.argv[1] == 'rnn':
        model = Model('rnn')
    else:
        raise ValueError('''Usage: python predict.py cnn/rnn''')
    with open(predict_dir, 'r', encoding='utf-8') as f:
        for news in f.readlines():
            print('\033[1;32;40m{0}\033[0m: {1} \n'.format(model.predict(news), news))
