#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午1:34
# @Author: guipeng
# @Version: 
# @File: example_journalism_tests.py
# @Contact: aceguipeng@gmail.com 
# @desc:　贝叶斯样例　新闻文本数据
'''
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from common import utils


def main():
    news = fetch_20newsgroups(subset='all')
    print(len(news.data))
    print(news.data[0])

    x_train, x_test, y_train, y_test = utils.prepare_train_and_test_sets(news.data, news.target,
                                                                         fit_model=CountVectorizer())
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(x_train, y_train)
    predict_result = multinomial_nb.predict(x_test)
    utils.get_train_model_score(multinomial_nb, x_test, y_test, predict_result)


if __name__ == '__main__':
    main()
