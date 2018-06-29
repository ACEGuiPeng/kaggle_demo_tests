#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午1:16
# @Author: guipeng
# @Version: 
# @File: example_load_digits.py
# @Contact: aceguipeng@gmail.com 
# @desc:　支持向量机分类器样例　手写数字图像识别
'''
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import utils


def main():
    digits_data = load_digits()
    x_train, x_test, y_train, y_test = utils.prepare_train_and_test_sets(digits_data.data, digits_data.target)

    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.transform(x_test)

    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    predict_result = linear_svc.predict(x_test)
    utils.get_train_model_score(linear_svc, x_test, y_test, predict_result)


if __name__ == '__main__':
    main()
