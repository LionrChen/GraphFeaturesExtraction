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
import pandas as pd
import numpy as np

from graph import Graph, SCC_graph


def list_to_heavy(array):
    return list(set(array))


# origin file
data_file = open("../dataset/digg_friends_format.csv", 'r')
read_file = data_file.readlines()

# # The file use to write extracted features.
# output_file = open("../dataset/digg_friends_features.csv", 'w')
# csv_write = csv.writer(output_file, dialect='excel')

graph = Graph()
vertices = set()
edges = []
for line in read_file:
    line = line.strip("\n").split(",")
    vertices.add(line[2])
    vertices.add(line[3])
    if line[0] is 1 or '1':
        edges.append((line[2], line[3]))

graph.add_nodes(vertices)
graph.add_edges(edges)


# pandas read csv
def get_indegree(node1):
    return graph.nodes.get(str(node1)).indegree


def get_nexts(node):
    return graph.nodes.get(str(node)).nexts


def get_subgraph(node1, node2):
    neib = [node1, node2]
    neib = list_to_heavy(neib + graph.nodes.get(node1).nexts + graph.nodes.get(node2).nexts)
    return graph.subgraph(neib)


feature_df = pd.DataFrame()

df = pd.read_csv("../dataset/digg_friends_format.csv", header=None)
feature_df[0] = df[0]
print("Start processing Indegree:")
feature_df[1] = df.apply(lambda row: get_indegree(row[2]), axis=1)  # indegree
print("Indegree Done.")
# print("Start calculate SCC score:")
# feature_df[2] = df.apply(lambda row: SCC_graph(get_subgraph(str(row[2]), str(row[3]))).__len__(), axis=1)  # SCC
# print("SCC Done.")
print("Start calculate total friends:")
feature_df[3] = df.apply(lambda row: graph.total_neighbors(str(row[2]), str(row[3])), axis=1)  # total-friends
print("Total friends Done.")
print("Start calculate Jaccard coefficient:")
feature_df[4] = df.apply(lambda row: graph.jaccard_coefficient(str(row[2]), str(row[3])), axis=1)  # Jaccard-coefficient
print("Jaccard coefficient Done.")
print("Start calculate preference attachment:")
feature_df[5] = df.apply(lambda row: graph.preference_attachment(str(row[2]), str(row[3])), axis=1)  # preference-attachment
print("Preference attachment Done.")
print("Start calculate friend measure:")
feature_df[2] = df.apply(lambda row: graph.friend_measure(str(row[2]), str(row[3])), axis=1)  # friend-measure
print("Friend measure Done.")
feature_df.to_csv("../dataset/digg_friends_features.csv", sep=',', float_format='%.2f', header=None, index=0)
# print(SCC_graph(subgraph))
# for line1 in read_file:
#     index += 1
#     print("Row {}".format(index))
#     line1 = line1.strip("\n").split(",")
#     # feature: is-friend, in-degree, SCC, total-friends, Jaccard-coefficient, preference-attachment, friend-measure
#     node1 = line1[2]
#     node2 = line1[3]
#     neib = [node1, node2]
#     neib = list_to_heavy(neib + graph.nodes.get(node1).nexts + graph.nodes.get(node2).nexts)
#     #print("them neighbors is", neib, graph.nodes.get(line1[2]).nexts, graph.nodes.get(line1[3]).nexts)
#     subgraph = graph.subgraph(neib)
#     print("Processing Node {} and {}.".format(node1, node2))
#     csv_write.writerow([int(line1[0]),
#                         graph.nodes.get(node1).indegree,
#                         SCC_graph(subgraph).__len__(),
#                         graph.total_neighbors(node1, node2),
#                         graph.jaccard_coefficient(node1, node2),
#                         graph.preference_attachment(node1, node2),
#                         graph.friend_measure(node1, node2)
#                         ])
#
# print("Write over.")
#
# def get_indegree(node1):
#     return graph.nodes.get(node1).indegree
