#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# @Time: 18-6-29 下午7:34
# @Author: guipeng
# @Version: 
# @File: example_titanic_pratice.py
# @Contact: aceguipeng@gmail.com 
# @desc:
'''
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier


def main():
    # 1 查看训练集和测试集的数据特征
    train_data = pandas.read_csv('data/train.csv')
    test_data = pandas.read_csv('data/test.csv')
    print(train_data.info())
    print(test_data.info())
    # 2 人工选取预测有效的特征
    selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    x_train = train_data[selected_features]
    x_test = test_data[selected_features]

    y_train = train_data['Survived']

    # 3 补充缺失值
    # 得知Embared特征惨在缺失值，需要补完
    print(x_train['Embarked'].value_counts())
    print(x_test['Embarked'].value_counts())

    # 对于类别型特征，使用出现频率最高的特征来填充，可以作为减少引入误差的方法之一
    x_train['Embarked'].fillna('S', inplace=True)
    x_test['Embarked'].fillna('S', inplace=True)

    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
    print(x_train.info())
    print(x_test.info())

    # 4 采用DictVectorizer对特征向量化
    dict_vectorizer = DictVectorizer(sparse=False)
    x_train = dict_vectorizer.fit_transform(x_train.to_dict(orient='record'))
    print(dict_vectorizer.feature_names_)
    x_test = dict_vectorizer.transform(x_test.to_dict(orient='record'))

    # 5 训练模型
    forest_classifier = RandomForestClassifier()
    xgb_classifier = XGBClassifier()

    # 使用5折交叉验证的方式进行性能评估
    forest_mean_score = cross_val_score(forest_classifier, x_train, y_train, cv=5).mean()
    print(forest_mean_score)
    xgb_mean_score = cross_val_score(xgb_classifier, x_train, y_train, cv=5).mean()
    print(xgb_mean_score)

    # 6 使用并行网格搜索的方式选择更好的超参组合
    params = {
        'max_depth': range(2, 8), 'n_estimators': range(100, 1200, 200),
        'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]
    }
    xgbc_best = XGBClassifier()
    grid_search_cv = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5)
    grid_search_cv.fit(x_train, y_train)
    print(grid_search_cv.best_score_)
    print(grid_search_cv.best_params_)

    # 7 预测结果并写入文件
    predict_result = grid_search_cv.predict(x_test)
    submission_data = pandas.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predict_result})
    submission_data.to_csv('data/submission/titanic_submission.csv', index=False)


if __name__ == '__main__':
    main()
