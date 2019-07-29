#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/26 19:08
# @Author  : ChenSir
# @File    : graphBase.py
# @Software: PyCharm

from node import Node
from edge import Edge


class GraphBase(object):
    def __init__(self, directed, weighted):
        self.nodes = dict()
        self.edges = []
        self.directed = directed
        self.weighted = weighted

    def add_node(self, value):
        try:
            new_node = Node(value)
            self.nodes[value] = new_node
            return new_node
        except:
            print("Invalid vertex id '{}', may be exits.".format(value))

    def __getitem__(self, item):
        if isinstance(item, int):
            # through up down mark gain item
            return self.nodes[item]
        elif isinstance(item, slice):
            # through slice gain item
            start = item.start
            stop = item.stop
            return [node for node in self.nodes[start:stop]]
        elif isinstance(item, str):
            # through name gain item
            return self.nodes.get(item, None)
        elif isinstance(item, list):
            # through list gain items
            temp_list = []
            for key in list:
                temp_list.append(self.nodes.get(key))
            return temp_list
        else:
            # key error
            raise TypeError('The type of key is error you input.')

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

    def add_edge(self, sour, terg):
        try:
            if not self.weighted:
                new_edge = Edge(False, sour, terg)
                self.edges.append(new_edge)
        except:
            print("The edge already exits.")

    def add_weight_edge(self, weight, sour, terg):
        try:
            if self.weighted:
                new_edge = Edge(weight, sour, terg)
                self.edges.append(new_edge)
        except:
            print("The edge already exits.")

    def add_edges(self, edges):
        pass

    def __iter__(self):
        print("Number of node: {}.".format(self.nodes.__len__()))
        print("Head 10 of node and neighbors: ")
        for index, (key,node) in enumerate(self.nodes.items()):
            print("Node {}: key is {}, it's neighbor are {}".format(node.value, key, node.nexts))
            if index is 9:
                break

    def subgraph(self, nodes):
        pass
