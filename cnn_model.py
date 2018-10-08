# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 16:48
# @Author  : Jason
# @FileName: cnn_model.py

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNConfig(object):
    """CNN配置参数"""
    embedding_size = 64  # 词向量维度,即每个词用64维的向量来表示
    seq_length = 600  # 序列长度，即每个文本的长度
    num_classes = 10  # 类别数目，所有文本可被分为不同的10类
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表大小，所有文本中出现的word的总个数

    hidden_size = 128  # 全连接层神经元数目

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 0.001  # 学习率

    batch_size = 64  # 每批训练大小，即一个iterator训练64个样本，并且更新一次参数
    num_epochs = 10  # 总迭代次数

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class CNN(object):
    """CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个需要输入的变量
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        self.cnn()

    def cnn(self):
        # 词向量映射  embedding layer
        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size,
                                                      self.config.embedding_size])  # 创建5000x64名为embedding的变量
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)  # 选取一个张量里面索引对应的元素

        with tf.name_scope('cnn'):
            # convolution layer
            conv = tf.layers.conv1d(inputs=embedding_inputs, filters=self.config.num_filters,
                                    kernel_size=self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, axis=1, name='gmp')  # axis=1维度上的最大值,即行上取最大值

        with tf.name_scope('score'):
            fc = tf.layers.dense(inputs=gmp, units=self.config.hidden_size, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.config.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(inputs=fc, units=self.config.num_classes, name='fc2')
            self.y_predict_class = tf.argmax(tf.nn.softmax(self.logits), axis=1)  # 返回ont-hot中最大值的索引

        with tf.name_scope('optimizer'):
            # 损失函数,交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, axis=1),
                                    self.y_predict_class)  # input_y为10x1的列向量,axis=1表示在行上取最大值
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
