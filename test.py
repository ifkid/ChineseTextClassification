# -*- coding: utf-8 -*-
# @Time    : 2018/9/22 18:57
# @Author  : Jason
# @FileName: test.py

import os

tensorboard_dir = 'tensorboard'


def model(mode_name):
    tensorboard_dir_model = os.path.join(tensorboard_dir, mode_name)
    if not os.path.exists(tensorboard_dir_model):
        os.makedirs(tensorboard_dir_model)


if __name__ == "__main__":
    model('cnn')
