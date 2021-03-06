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

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_model_name(model):
    return model.__class__.__name__


def get_random_seed():
    random_num = random.randint(1, 10000)
    print('random int: {}'.format(random_num))
    return random_num


def prepare_train_and_test_sets(x_data, y_data, random_state=get_random_seed(), test_percent=0.25):
    """准备训练数据和测试数据

    :param x_data:  输入数据
    :param y_data:　输出数据
    :param random_state:    随机数种子
    :param test_percent:    测试数据占比
    :return:　拆分的训练集和测试集
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_percent,
                                                        random_state=random_state)

    return x_train, x_test, y_train, y_test


def get_train_model_prediction(trained_model, x_train, y_train, x_test):
    trained_model.fit(x_train, y_train)
    predict_result = trained_model.predict(x_test)
    return trained_model, predict_result


def get_train_model_score(trained_model, x_test, y_test, predict_result):
    """获取模型的训练评估结果"""
    model_name = get_model_name(trained_model)
    print('Accuracy of {}: {}'.format(model_name, trained_model.score(x_test, y_test)))
    print('Precision of {}:\n {}'.format(model_name, classification_report(y_test, predict_result)))


def get_trained_result(training_model, x_test, x_train, y_test, y_train):
    trained_model, predict_result = get_train_model_prediction(training_model, x_train, y_train, x_test)
    get_train_model_score(trained_model, x_test, y_test, predict_result)
