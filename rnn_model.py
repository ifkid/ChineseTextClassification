# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 14:01
# @Author  : Jason
# @FileName: rnn_model.py

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.layers import dense
from tensorflow.nn import relu


class RNNConfig(object):
    """RNN配置参数"""
    embedding_size = 64  # 词向量维度
    seq_length = 600  # 序列长度，即每个文本的长度
    num_classes = 10  # 类别数目，所有文本可被分为不同的10类
    vocab_size = 5000  # 词汇表大小，所有文本中出现的word的总个数

    num_layers = 2  # 隐藏层层数
    hidden_size = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm or gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 0.001  # 学习率

    batch_size = 128  # 每批训练大小，即一个iterator训练64个样本，并且更新一次参数
    num_epochs = 10  # 总迭代次数

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class RNN(object):
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.rnn()

    def rnn(self):
        """RNN模型"""

        def lstm_cell():  # lstm核
            return BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)

        def gru_cell():  # gru核
            return GRUCell(self.config.hidden_size)

        def dropout():  # 在每一个rnn核后面加一个dropout层
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return DropoutWrapper(cell=cell, output_keep_prob=self.config.dropout_keep_prob)

        # 词向量映射
        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = MultiRNNCell(cells=cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序作为输出结果

        with tf.name_scope('score'):
            # 全连接层,后面连接dropout以及relu激活
            fc = dense(last, self.config.hidden_size, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.config.dropout_keep_prob)
            fc = relu(fc)

            # 分类器
            self.logits = dense(fc, self.config.num_classes, name='fc2')
            self.y_predict_class = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('optimize'):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_predict_class)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
