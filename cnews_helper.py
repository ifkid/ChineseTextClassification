# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 19:31
# @Author  : Jason
# @FileName: cnews_helper.py

import sys
import time
from collections import Counter
from datetime import timedelta
from importlib import reload

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    is_py3 = False
    sys.setdefaultencoding('utf-8')


def open_file(file, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(file, mode=mode, encoding='utf-8', errors='ignore')
    else:
        return open(file, mode=mode)


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def read_file(filename):
    """读取文本文件"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    labels.append(native_content(label))
                    contents.append(list(native_content(content)))
                else:
                    continue
            except Exception:
                raise Exception
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size):
    data, _ = read_file(train_dir)
    all_data = []
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)
    countet_pairs = counter.most_common(vocab_size - 1)  # 从大到小排列
    words, _ = list(zip(*countet_pairs))  # 解压得到出现频率从大到小的word
    # 添加一个<PAD>将所有文本pad成同一长度
    words = ['<PAD>'] + list(words)
    # 'seq'.join(),将字符串、元组、列表中的元素以指定的字符(seq)连接生成一个新的字符串
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_categories():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """"""
    contents, labels = read_file(filename=filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x]
                        for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # print("data_id: \n", data_id)
    # print("label_id: \n", label_id)

    # 使用keras提供的sequences将文本pad成同一长度
    x_pad = kr.preprocessing.sequence.pad_sequences(sequences=data_id, maxlen=max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    # print("x_pad: \n", x_pad)
    # print("y_pad: \n", y_pad)
    return x_pad, y_pad


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    iterators = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(data_len)
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(iterators):
        start_indice = i * batch_size
        end_indice = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_indice:end_indice], y_shuffle[start_indice:end_indice]


def feed_data(model, x_batch_train, y_batch_train, dropout_keep_prob):
    feed_dict = {
        model.input_x: x_batch_train,
        model.input_y: y_batch_train,
        model.dropout_keep_prob: dropout_keep_prob
    }
    return feed_dict
