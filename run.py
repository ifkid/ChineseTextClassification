# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 16:44
# @Author  : Jason
# @FileName: run.py

import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics

from Text.cnews_helper import build_vocab, read_vocab, read_categories, process_file, get_time_dif, batch_iter, \
    feed_data
from Text.cnn_model import CNNConfig, CNN
from Text.rnn_model import RNNConfig, RNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

tensorboard_dir = 'tensorboard'
save_dir = 'checkpoint'


# save_path_cnn = os.path.join(save_dir_cnn, 'best_validation')  # 最佳验证结果保存路径


def evaluate(session, x_val, y_val):
    data_len = len(x_val)
    batch_val = batch_iter(x_val, y_val, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_val:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = session.run([model.loss, model.accuracy], feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train(model_name):
    print("Configuring tensorboard and saver...")
    tensorboard_dir_model = os.path.join(tensorboard_dir, model_name)
    if not os.path.exists(tensorboard_dir_model):
        os.makedirs(tensorboard_dir_model)

    # 配置tensorboard
    tf.summary.scalar(name='loss', tensor=model.loss)
    tf.summary.scalar(name='accuracy', tensor=model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir_model)

    # 配置saver
    save_dir_model = os.path.join(save_dir, model_name)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)
    print("Finish to configure tensorboard and saver successfully. \n")
    print('Loading training and validation data...')
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id=word_to_id, cat_to_id=cat_to_id, max_length=config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id=word_to_id, cat_to_id=cat_to_id, max_length=config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Finish to load training and validation data successfully. \n")
    print("Time usage: ", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch: {}".format(epoch + 1))
        batch_train = batch_iter(x_train, y_train, batch_size=config.batch_size)
        for x_batch_train, y_batch_train in batch_train:
            feed_dict = feed_data(model, x_batch_train, y_batch_train, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每save_per_batch轮次写入tensorboard
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每print_per_batch轮次输出一次训练集和验证集上的性能
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, accuracy_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, accuracy_val = evaluate(session, x_val, y_val)

                if accuracy_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = accuracy_val
                    last_improved = total_batch
                    save_path_model = os.path.join(save_dir_model, 'best_validation')
                    saver.save(sess=session, save_path=save_path_model)
                    improved_str = '*'
                else:
                    improved_str = ''
                time_dif = get_time_dif(start_time)
                msg = 'iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(
                    msg.format(total_batch, loss_train, accuracy_train, loss_val, accuracy_val, time_dif, improved_str))
            # 优化
            session.run(model.optimizer, feed_dict=feed_dict)
            total_batch += 1

            # 如果验证集正确率长期得不到提升， 则提前结束训练
            if total_batch - last_improved > require_improvement:
                print("No optimized for a long time, stop training automatically.")
                flag = True
                break
        if flag:
            break


def test(model_name):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
    print("Finish to load test data successfully. \n")
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 读取保存的模型
    save_dir_model = os.path.join(save_dir, model_name)
    save_path_model = os.path.join(save_dir_model, 'best_validation')
    saver.restore(sess=session, save_path=save_path_model)

    print("Testing...")
    loss_test, acc_test = evaluate(session, x_test, y_test)

    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_class = np.argmax(y_test, axis=1)
    y_predict_class = np.zeros(shape=len(x_test), dtype=np.int32)
    for i in range(num_batch):
        start_indices = i * batch_size
        end_indices = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_indices:end_indices],
            model.dropout_keep_prob: 1.0
        }
        y_predict_class[start_indices:end_indices] = session.run(model.y_predict_class, feed_dict)
    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_class, y_predict_class, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_class, y_predict_class)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ['cnn', 'rnn'] or sys.argv[2] not in ['train', 'test']:
        raise ValueError("""Usage: python run.py cnn/rnn train/test""")
    elif sys.argv[1] == "cnn":
        print('Configuring CNN model...')
        config = CNNConfig()
        print('Finish to configure CNN model successfully. \n')
        if not os.path.exists(vocab_dir):
            print('Building the vocab...')
            build_vocab(train_dir, vocab_dir, config.vocab_size)
            print('Finish to build vocab successfully. \n')
        else:
            pass
        categories, cat_to_id = read_categories()
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        model = CNN(config=config)
        if sys.argv[2] == "train":
            train("cnn")
        elif sys.argv[2] == "test":
            test("cnn")
        else:
            raise ValueError("""Usage: python run.py cnn/rnn train/test""")
    elif sys.argv[1] == "rnn":
        print("Configuring RNN model...")
        config = RNNConfig()
        print("Finish to configure RNN model successfully. \n")
        if not os.path.exists(vocab_dir):
            print('Building the vocab...')
            build_vocab(train_dir, vocab_dir, config.vocab_size)
            print('Finish to build vocab successfully. \n')
        else:
            pass
        categories, cat_to_id = read_categories()
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        model = RNN(config)
        if sys.argv[2] == 'train':
            train("rnn")
        elif sys.argv[2] == "test":
            test("rnn")
        else:
            raise ValueError("""Usage: python run.py cnn/rnn train/test""")
