#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午4:14
# @Author: guipeng
# @Version: 
# @File: example_load_iris.py
# @Contact: aceguipeng@gmail.com 
# @desc:
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import utils


def main():
    iris = load_iris()
    print(iris.DESCR)

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25,
                                                        random_state=utils.get_random_seed())

    k_neighbors_classifier, model_prediction = utils.get_train_model_prediction(KNeighborsClassifier(), x_train,
                                                                                y_train, x_test)
    utils.get_train_model_score(k_neighbors_classifier, x_test, y_test, model_prediction)


if __name__ == '__main__':
    main()
