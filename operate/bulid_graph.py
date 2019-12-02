#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 15:40
# @Author  : ChenSir
# @File    : bulid_graph.py
# @Software: PyCharm

import networkx as nx
import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool


def fun(graph, index, node, nbr, weight):
    # feature: is-friend, in-degree, SCC, total-friends, Jaccard-coefficient, preference-attachment, friend-measure
    in_d = in_degree(graph, node, nbr)
    SCC = scc(graph, node, nbr)
    tf = total_friends(graph, node, nbr)
    j_c = jc(graph, node, nbr)
    p_a = pa(graph, node, nbr)
    f_m = fm(graph, node, nbr)
    print("Row {} has processed.\n".format(index))
    return [weight, in_d, SCC, tf, j_c, p_a, f_m]


def write_to_csv(msg):
    csv_write.writerow(msg)


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


def in_degree(graph, node, nbr):
    return graph.degree(nbr)


def total_friends(graph, node, nbr):
    return len(set(list(graph.neighbors(node)) + list(graph.neighbors(nbr))))


def scc(graph, node, nbr):
    subgraph_list = list(set(list(graph.neighbors(node)) + list(graph.neighbors(nbr))))
    return nx.number_strongly_connected_components(graph.subgraph(subgraph_list).to_directed())


def jc(graph, node, nbr):
    for u, v, jc in nx.jaccard_coefficient(graph, [(node, nbr)]):
        return jc


def pa(graph, node, nbr):
    for u, v, PA in nx.preferential_attachment(graph, [(node, nbr)]):
        return PA


def fm(graph, node, nbr):
    nbrs_node1 = list(graph.neighbors(node))
    nbrs_node2 = list(graph.neighbors(nbr))
    F_m = 0
    for node1 in nbrs_node1:
        if node1 is not nbr:
            nbrs = set(graph.neighbors(node1))
            nbrs.add(node1)
            F_m += len(set(nbrs_node2).difference(nbrs))
    return F_m


if __name__ == '__main__':
    data_file_pd = pd.read_csv("../operate/sample_graph.csv", usecols=(0, 1, 2), header=None)
    # The file use to write extracted features.
    output_file = open("blog_data_second.csv", 'w')
    csv_write = csv.writer(output_file, dialect='excel')

    vertices = list(set((data_file_pd.iloc[:, 0]).tolist() + (data_file_pd.iloc[:, 1]).tolist()))
    edges = []

    for index, row in data_file_pd.iterrows():
        if row[2]:
            edges.append((row[0], row[1], {'weight': row[2]}))

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    print("number of edges: {}".format(graph.number_of_edges()))
    # pool = Pool(6)
    for index, row in data_file_pd.iterrows():
        msg = fun(graph, index, row[0], row[1], row[2])
        write_to_csv(msg)
        # pool.apply_async(fun, (graph, index, row[0], row[1], row[2]), callback=write_to_csv)

    # pool.close()
    # pool.join()
    output_file.close()

    print("Write over.")
