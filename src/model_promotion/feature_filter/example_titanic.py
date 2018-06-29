#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午6:45
# @Author: guipeng
# @Version: 
# @File: example_titanic.py
# @Contact: aceguipeng@gmail.com 
# @desc: 实用Titanic数据集，通过特征筛选提升决策树性能
'''
import numpy
import pandas
from matplotlib import pylab
from sklearn import feature_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from common import utils


def main():
    titanic = pandas.read_csv('dataset/titanic.csv')

    # 分离数据特征与预测目标
    x_set = titanic.drop(['row.names', 'name', 'survived'], axis=1)
    y_set = titanic['survived']
    x_set.fillna(x_set['age'].mean(), inplace=True)

    x_train, x_test, y_train, y_test = utils.prepare_train_and_test_sets(x_set, y_set)

    # 类别型特征向量化
    dict_vectorizer = DictVectorizer()
    x_train = dict_vectorizer.fit_transform(x_train.to_dict(orient='record'))
    x_test = dict_vectorizer.transform(x_test.to_dict(orient='record'))
    print(dict_vectorizer.feature_names_)
    print("=" * 100)

    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
    utils.get_trained_result(decision_tree_classifier, x_test, x_train, y_test, y_train)

    # 筛选前20%的特征
    select_percentile = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
    x_train_fs = select_percentile.fit_transform(x_train, y_train)
    x_test_fs = select_percentile.transform(x_test)
    decision_tree_classifier.fit(x_train_fs, y_train)
    print(decision_tree_classifier.score(x_test_fs, y_test))

    # 通过交叉验证，按照固定间隔的百分比筛选特征，并作图展现性能随特征筛选比例的变化
    percentiles = range(1, 100, 2)
    results = []

    for i in percentiles:
        new_fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
        x_train_fs = new_fs.fit_transform(x_train, y_train)
        scores = cross_val_score(decision_tree_classifier, x_train_fs, y_train, cv=5)
        results = numpy.append(results, scores.mean())

    print(results)

    # 找到提现最佳性能的特征筛选的百分比
    opt = numpy.where(results == results.max())[0]
    best_percentile = percentiles[opt[0]]
    print('Optimal number of features: {}'.format(best_percentile))

    # 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
    pylab.plot(percentiles, results)
    pylab.xlabel('percentiles of feature')
    pylab.ylabel('accuracy')
    pylab.show()

    best_fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=best_percentile)
    x_train_fs = best_fs.fit_transform(x_train, y_train)
    decision_tree_classifier.fit(x_train_fs, y_train)
    x_test_fs = best_fs.transform(x_test)
    print('new score: {}'.format(decision_tree_classifier.score(x_test_fs, y_test)))


if __name__ == '__main__':
    main()
