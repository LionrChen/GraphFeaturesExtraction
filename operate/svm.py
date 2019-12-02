#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/10 12:29
# @Author  : ChenSir
# @File    : svm.py
# @Software: PyCharm

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer,OneHotEncoder,MinMaxScaler,MaxAbsScaler,StandardScaler,Normalizer
import pandas as pd

data = pd.read_csv("../operate/blog_data_second.csv", header=None)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit_transform(X_train)
# min_max_scaler.fit_transform(X_test)
normalize = Normalizer(norm='l2')
normalize.transform(X_train)
normalize.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(X_train, y_train)


print('Accuracy score: {}'.format(accuracy_score(y_test, clf.predict(X_test))))
