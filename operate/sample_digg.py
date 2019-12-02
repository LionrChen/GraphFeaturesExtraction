#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 22:09
# @Author  : ChenSir
# @File    : sample_digg.py
# @Software: PyCharm
from sample.RandomWalk.RandWalk import RandomWalkSample
import pandas as pd

if __name__ == '__main__':
    rand_walk = RandomWalkSample(10000)
    data_file = pd.read_csv("../dataset/digg_friends_format.csv", usecols=(0, 2, 3))
    data_file.sort_index(axis=1)
    rand_walk.build_graph(data_file)
    rand_walk.start_rand_walk()
    rand_walk.graph_to_csv(rand_walk.sample_graph)
