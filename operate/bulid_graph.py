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


data_file_pd = pd.read_csv("../dataset/testdata.csv", usecols=(0, 2, 3), header=None)
data_file_pd.apply(convert_currency)
# The file use to write extracted features.
output_file = open("../dataset/digg_friends_features.csv", 'w')
csv_write = csv.writer(output_file, dialect='excel')

graph = ig.Graph()
vertices = []
edges = []

nodes1 = data_file_pd[2].tolist()
nodes2 = data_file_pd[3].tolist()
# print(list(sorted(set(nodes1+nodes2))))
vertices = list(sorted(set(nodes1+nodes2)))
# for line in csv_read_file:
#     if line[2] not in vertices:
#         vertices.append(line[2])
#     if line[3] not in vertices:
#         vertices.append(line[3])
#     if line[0] is 1 or '1':
#         edges.append((line[2], line[3]))

for index, row in data_file_pd.iterrows():
    if row[0] is 1 or '1':
        edges.append((row[2], row[3]))

graph.add_vertices(vertices)
graph.add_edges(edges)
numbers = graph.indegree()
print(numbers)
graph.write_svg("tempSVG.svg")
