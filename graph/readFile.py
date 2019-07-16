#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ChenSir'
__mtime__ = '2019/7/16'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
            ┃       ┃
            ┃ ┳┛ ┗┳ ┃
            ┃   ┻   ┃
            ┗━┓   ┏━┛
              ┃   ┗━━━┓
              ┃神兽保佑 ┣┓
              ┃ 永无BUG！ ┏┛
              ┗┓┓┏━━┳┓┏┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
"""

from graph import Graph

data_file = open("../dataset/soc-sign-bitcoinotc.csv", 'r')
fc = data_file.readlines()
graph = Graph()
for line in fc:
    data_f = line.split(",")
    # print("Source: {}, Target: {}".format(data_f[0],data_f[1]))
    graph.add_node(data_f[0])
    graph.add_node(data_f[1])
    graph.add_edge(data_f[0],data_f[1])
graph.__iter__()