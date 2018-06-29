#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time: 18-6-29 上午11:17
# @Author: guipeng
# @Version:
# @File: example_tumour_classifiers.py
# @Contact: aceguipeng@gmail.com
# @desc:
"""
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from common import utils
from common.utils import prepare_train_and_test_sets

"""
LogisticRegression比起SGDClassifier在测试集上表现有更高的准确性（Accuracy）。这是因为Scikit-learn中采用解析的方式精确计算
LogisticRegression的参数，而使用梯度法估计SGDClassifier的参数。

线性分类器可以说是最为基本和常用的机器学习模型。尽管其受限于数据特征与分类目标之间的线性假设，我们仍然可以在科学研究与工程实践
中把线性分类器的表现性能作为基准。这里所使用的模型包括LogisticRegression与SGDClassifier。相比之下，前者对参数的计算采用精确解
析的方式，计算时间长但是模型性能略高；后者采用随机梯度上升算法估计模型参数，计算时间短但是产出的模型性能略低。一般而言，对于训练
数据规模在10万量级以上的数据，考虑到时间的耗用，更加推荐使用随机梯度算法对模型参数进行估计。

"""

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of CellShape',
                'Marginal Adhesion', 'Single Epithelial CellSize', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses', 'Class']
cn_column_names = ['样本编号', '肿块厚度', '细胞大小均匀性', '细胞形状均匀性', '边缘粘附', '单上皮细胞大小', '裸核',
                   '乏味染色体', '正常核', '有丝分裂', '肿瘤类型']

INNOCENT_TUMOUR = 2  # 良性肿瘤
MALIGNANT_TUMOR = 4  # 恶性肿瘤


def preprocess_data():
    """数据预处理

    :return:
    """
    data_file = 'datasets/breast-cancer-wisconsin.data.csv'
    data = pandas.read_csv(data_file, names=cn_column_names)
    print(data)
    # 替换异常值为缺失值na
    data = data.replace(to_replace='?', value=numpy.nan)
    # 丢弃缺失值的数据
    data = data.dropna(how='any')
    data_shape = data.shape
    print('data sum: {},column num: {}'.format(data_shape[0], data_shape[1]))
    print("=" * 100)
    return data


def main():
    """
        # 1 预处理数据
        # 2 拆分数据集为训练集和测试集
        # 3 训练模型
        # 4 评估模型
    :return:
    """
    processed_data = preprocess_data()
    x_train, x_test, y_train, y_test = prepare_train_and_test_sets(processed_data[cn_column_names[1:10]],
                                                                   processed_data[cn_column_names[10]],
                                                                   )
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    lr_predict_result = logistic_regression.predict(x_test)
    utils.get_train_score(logistic_regression, x_test, y_test, lr_predict_result)

    sgd_classifier = SGDClassifier()
    sgd_classifier.fit(x_train, y_train)
    sgdc_predict_result = sgd_classifier.predict(x_test)
    utils.get_train_score(sgd_classifier, x_test, y_test, sgdc_predict_result)


if __name__ == '__main__':
    main()
