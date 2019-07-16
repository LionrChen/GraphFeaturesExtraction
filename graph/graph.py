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
from node import Node

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, id):
        if id not in self.nodes.keys():
            new_node = Node(id)
            self.nodes[id] = new_node
            return new_node
        else:
            print("Node already exits.")

    def disable_all(self):
        """ Disable all nodes in graph
            Useful for switching on small subnetworks
            of bayesian nets
        """
        for k, v in iter(self.nodes):
            v.disable()

    def reset(self):
        """ Reset messages to original state
        """
        for k, v in iter(self.nodes):
            v.reset()

    def add_edge(self, sor, terg):
        if terg not in self.nodes[sor].neighbors:
            self.nodes[sor].neighbors.append(terg)
            print("Add edge to node {}, e: {} -> {}".format(sor,sor,terg))
        else:
            print("The edge already exits.")

    def __iter__(self):
        print("Number of node: {}.".format(self.nodes.__len__()))