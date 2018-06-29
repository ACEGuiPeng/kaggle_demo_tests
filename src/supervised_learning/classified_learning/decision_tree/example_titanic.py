#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午4:30
# @Author: guipeng
# @Version: 
# @File: example_titanic_2.py
# @Contact: aceguipeng@gmail.com 
# @desc:
'''
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from common import utils


def main():
    titanic = pandas.read_csv('dataset/titanic.csv')

    x_set = titanic[['pclass', 'age', 'sex']]
    y_set = titanic['survived']
    x_set.fillna(x_set['age'].mean(), inplace=True)
    x_train, x_test, y_train, y_test = utils.prepare_train_and_test_sets(x_set, y_set)

    dict_vectorizer = DictVectorizer(sparse=False)
    x_train = dict_vectorizer.fit_transform(x_train.to_dict(orient='record'))
    x_test = dict_vectorizer.fit_transform(x_test.to_dict(orient='record'))

    decision_tree_classifier = DecisionTreeClassifier()
    utils.get_trained_result(decision_tree_classifier, x_test, x_train, y_test, y_train)


if __name__ == '__main__':
    main()
