#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 15:40
# @Author  : ChenSir
# @File    : bulid_graph.py
# @Software: PyCharm

import igraph as ig
import csv
import numpy as np
import pandas as pd
from SCC import SCC


# graph = ig.Graph.Read_Edgelist("../dataset/digg_friends.csv", directed=False)
# neighbors = graph.neighborhood()
# print(type(neighbors))


# origin file
# data_file = open("../dataset/testdata.csv", 'r')
# csv_read_file = csv.reader(data_file)

# read file use numpy
# data_file_np = np.loadtxt("../dataset/testdata.csv", delimiter=",", usecols=(0, 2, 3))


# read file use pandas
def convert_currency(value):
    """
    convert chars to float
    if failed return 0
    """
    try:
        return np.float(value)
    except Exception:
        return 0


data_file_pd = pd.read_csv("../dataset/digg_friends_format.csv", usecols=(0, 2, 3), header=None)
# data_file_pd.apply(convert_currency)
# The file use to write extracted features.
output_file = open("../dataset/digg_friends_features.csv", 'w')
csv_write = csv.writer(output_file, dialect='excel')

vertices = set()
edges = []

for index, row in data_file_pd.iterrows():
    edges.append((row[2], row[3]))


# graph.add_vertices(list(vertices))
# graph.add_edges(edges)
graph = ig.Graph.TupleList(edges,directed=False, vertex_name_attr='name', edge_attrs=None, weights=False)
# SCCs = {}
# for ver in graph.vs:
#     subgraph = graph.subgraph(graph.neighbors(ver))
#     temp_graph = {}
#     for p in subgraph.vs:
#         temp_graph[p["name"]] = subgraph.neighbors(p)
#     for key in temp_graph.keys():
#         nei = []
#         for id in temp_graph[key]:
#             nei.append(subgraph.vs[id]["name"])
#         temp_graph[key] = nei
#     print(temp_graph)
#     SCCs[ver["name"]] = SCC(temp_graph).__len__()
numbers = graph.indegree()
for index, row in data_file_pd.iterrows():
    # feature: is-friend,in-degree,
    csv_write.writerow([row[0], numbers[row[3]]])
graph.neighbors(5)
vertic = graph.get
graph.subgraph(5)
print("Write over.")
