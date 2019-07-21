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
import csv

from graph import Graph
# origin file
data_file = open("../dataset/digg_friends.csv", 'r')
csv_read_file = csv.reader(data_file)

# The file use to write extracted features.
output_file = open("../dataset/digg_friends_features.csv", 'w')
csv_write = csv.writer(output_file, dialect='excel')

graph = Graph()

for line in csv_read_file:
    graph.add_node(line[2])
    graph.add_node(line[3])
    if line[0] is 1 or '1':
        graph.add_edge(line[2], line[3])

graph.__iter__()

for line1 in csv_read_file:
    # feature: is-friend,in-degree,
    csv_write.writerow([int(line1[0]), graph.nodes.get(line1[2]).neighbors.__len__()])
    print([int(line1[0]), graph.nodes.get(line1[2]).neighbors.__len__()])

print("Write over.")
