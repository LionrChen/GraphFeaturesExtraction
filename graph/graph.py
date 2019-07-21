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
        self.nodes = dict()

    def add_node(self, id):
        if id not in self.nodes.keys():
            new_node = Node(id)
            self.nodes[id] = new_node
            return new_node
        else:
            print("Node already exits.")

    def add_node_object(self, node):
        if node not in self.nodes:
            self.nodes[node.node_id] = node
            return node
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

    def add_edge(self, sour, terg):
        if terg not in self.nodes[sour].neighbors:
            self.nodes[sour].neighbors.append(terg)
            print("Add edge to node {}, e: {} -> {}".format(sour, sour, terg))
        else:
            print("The edge already exits.")

    def __iter__(self):
        print("Number of node: {}.".format(self.nodes.__len__()))


def SCC(sour, targ, graph):
    """ calculate the number of strongly connected components within neighborhood-subgraph.
        First generate 1-dimensional neighborhood-subgraph with source and target node.
    """
    neighborhood_graph = graph()
    neighborhood_graph.add_node(sour)
    neighborhood_graph.add_node(targ)
    # add nodes
    for node in graph[sour].neighbors:
        if node not in neighborhood_graph.nodes:
            neighborhood_graph.add_node_object(node)
    for node in graph[targ].neighbors:
        if node not in neighborhood_graph.nodes:
            neighborhood_graph.add_node_object(node)
    # add edges, check out each node from source graph to estimate whether exits edge among them.
    for node in neighborhood_graph.nodes.keys():
        for nei in neighborhood_graph[node].neighbors:
            if nei not in neighborhood_graph.nodes.keys():
                del nei

    # reverse graph
    def reverse_graph(subgraph):
        re_graph = dict()
        for key in subgraph.keys():
            re_graph[key] = subgraph.get(key, set())
        # reverse edge
        for key in subgraph.keys():
            for nei in subgraph[key]:
                re_graph[nei].neighbors.add(key)
        return re_graph

    # gain a sequence sorted by time
    def topo_sort(subgraph):
        res = []
        S = set()

        # Depth-first traversal/search
        def dfs(subgraph, u):
            if u in S:
                return
            S.add(u)
            for v in subgraph[u].neighbors:
                if v in S:
                    continue
                dfs(subgraph, v)
            res.append(u)

        # check each node was leave out
        for u in subgraph.keys():
            dfs(subgraph, u)

        res.reverse()
        return res

    # gain singe strongly connected components with assigns start node
    def walk(subgraph, s, S=None):
        if S is None:
            s = set()
        Q = []
        P = dict()
        Q.append(s)
        P[s] = None
        while Q:
            u = Q.pop()
            for v in subgraph[u].neighbors:
                if v in P.keys() or v in S:
                    continue
                Q.append(v)
                P[v] = P.get(v, u)
        return P

    # use to record the node of strongly connected components
    seen = set()
    # to store scc
    scc = []
    GT = reverse_graph(neighborhood_graph)
    for u in topo_sort(neighborhood_graph):
        if u in seen:
            continue
        C = walk(GT, u, seen)
        seen.update(C)
        scc.append(sorted(list(C.keys())))

    print(scc)
