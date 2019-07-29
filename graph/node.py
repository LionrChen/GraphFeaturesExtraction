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


class Node(object):
    def __init__(self, value):
        self.enabled = True
        self.value = value
        self.indegree = 0
        self.outdegree = 0
        self.nexts = []
        self.edges = []
        # self.incoming = []
        # self.outgoing = []
        # self.oldoutgoing = []

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
        for n in self.nexts:
            # don't call enable() as it will recursively enable entire graph
            n.enabled = True

    def neighbors(self):
        return self.nexts

    def neighborhood(self):
        return self.nexts + [self.value]

    def degree(self):
        return self.indegree + self.outdegree

    def indegree(self):
        return self.indegree

    def outdegree(self):
        return self.outdegree

    def update(self, indegree, outdegree, nexts, edges):
        self.indegee = indegree
        self.outdegree = outdegree
        self.nexts = nexts
        self.edges = edges
