#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 17:32
# @Author  : ChenSir
# @File    : deep_walk.py
# @Software: PyCharm

import math
import networkx as nx
import random
import numpy as np


def load_edgelist(filename):
    graph_format = 'edgelist'
    if graph_format == 'edgelist':
        g = nx.read_edgelist(filename)
        for edge in g.edges():
            u, v = edge[0], edge[1]
            g[u][v]['weight'] = 1.0

    return g


def deepwalk_walk(g, walk_length=40, start_node=None):
    walks = [start_node]
    while len(walks) < walk_length:
        cur = walks[-1]
        cur_nbs = list(g.neighbors(cur))
        if len(cur_nbs) > 0:
            walks.append(random.choice(cur_nbs))
        else:
            raise ValueError('node with 0 in_degree')
    return walks


def sample_walks(g, walk_length=40, number_walks=10, shuffle=True):
    total_walks = []
    print('Start sampling walks:')
    nodes = list(g.nodes())
    for iter_ in range(number_walks):
        print('\t iter:{} / {}'.format(iter_ + 1, number_walks))
        if shuffle:
            random.shuffle(nodes)
        for node in nodes:
            total_walks.append(deepwalk_walk(g, walk_length, node))
    return total_walks
