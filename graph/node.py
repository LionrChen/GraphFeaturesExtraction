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
    def __init__(self, id):
        self.enabled = True
        self.node_id = id
        self.neighbors = []
        # self.incoming = []
        # self.outgoing = []
        # self.oldoutgoing = []

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
        for n in self.neighbors:
            # don't call enable() as it will recursively enable entire graph
            n.enabled = True

    def get_neighbor_node(self):
        return self.neighbors

    def get_node_id(self):
        return self.node_id