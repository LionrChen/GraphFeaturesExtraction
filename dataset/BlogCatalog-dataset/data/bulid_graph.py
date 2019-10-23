#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 15:40
# @Author  : ChenSir
# @File    : bulid_graph.py
# @Software: PyCharm

import igraph as ig
import networkx as nx
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


data_file_pd = pd.read_csv("edges.csv", usecols=(0, 1), header=None)
nodes = pd.read_csv("nodes.csv", header=None)
# data_file_pd.apply(convert_currency)
# The file use to write extracted features.
output_file = open("blog_data.csv", 'w')
csv_write = csv.writer(output_file, dialect='excel')

vertices = (nodes.iloc[:, 0]).tolist()
edges = []

for index, row in data_file_pd.iterrows():
    edges.append((row[0], row[1]))

# graph.add_vertices(list(vertices))
# graph.add_edges(edges)
graph = nx.Graph()
graph.add_nodes_from(vertices)
print("Vertices: {}, and number is {}".format(vertices[:3], vertices.__len__()))
graph.add_edges_from(edges)
print("edges: {}, and number is {}".format(edges[:3], edges.__len__()))
print("number of edges: {}".format(graph.number_of_edges()))
total_edges = graph.number_of_edges()
processed_edges = 0
for node, nbrs in graph.adjacency():
    for nbr in nbrs:
        # F_m = 0
        # for nb_nbr in graph.neighbors(nbr):
        #     if nb_nbr not in nbrs and not graph.edges([nbr, nb_nbr]):
        #         F_m += 1
        # feature: is-friend, in-degree, SCC, total-friends, Jaccard-coefficient, preference-attachment, friend-measure
        # ,
        #                                len(set(graph.neighbors(node) + graph.neighbors(nbr))), nx.jaccard_coefficient(graph, [node, nbr]),
        #                                nx.preferential_attachment(graph, [node, nbr]), F_m
        subgraph = list(graph.neighbors(node)) + list(graph.neighbors(nbr))
        try:
            csv_write.writerow([1, nx.degree(graph, node), nx.number_strongly_connected_components(graph.subgraph(subgraph).to_directed())])
            processed_edges += 1
            print("{}: Node {} and {} has processed. {} edges the remaining.".format(processed_edges, node, nbr, total_edges-processed_edges))
        except:
            print("Node {} has generate exception.".format(node))
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
# numbers = graph.indegree()
# print(numbers)
# for index, row in data_file_pd.iterrows():
#     # feature: is-friend,in-degree,
#     try:
#         csv_write.writerow([row[0], numbers[index]])
#     except:
#         print("list index {} out of range.".format(index))
# print(graph.neighbors(5))
# print(graph.subgraph(5))
print("Write over.")
