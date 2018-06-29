#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午12:58
# @Author: guipeng
# @Version: 
# @File: utils.py
# @Contact: aceguipeng@gmail.com 
# @desc:
'''
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_model_name(model):
    return model.__class__.__name__


def get_random_seed():
    random_num = random.randint(1, 10000)
    print('random int: {}'.format(random_num))
    return random_num


def prepare_train_and_test_sets(x_data, y_data, random_state, test_percent=0.25):
    """准备训练数据和测试数据

    :param x_data:  输入数据
    :param y_data:　输出数据
    :param random_state:    随机数种子
    :param test_percent:    测试数据占比
    :return:　拆分的训练集和测试集
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_percent,
                                                        random_state=random_state)

    print("训练结果样本数量和类别分布:\n{}".format(y_train.value_counts()))
    print("=" * 100)
    print("测试结果样本数量和类别分布:\n{}".format(y_test.value_counts()))
    print("=" * 100)

    # 标准化数据
    # 保证每个维度的特征数据方差为1,均值为0。使得预测结果不会
    # 被某些维度过大的特征值而主导
    standard_scaler = StandardScaler()
    standard_x_train = standard_scaler.fit_transform(x_train)
    standard_x_test = standard_scaler.fit_transform(x_test)

    return standard_x_train, standard_x_test, y_train, y_test
