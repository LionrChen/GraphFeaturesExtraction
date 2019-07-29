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
from graphBase import GraphBase
from node import Node
from edge import Edge


class Graph(GraphBase):
    def __init__(self, directed=False, weighted=False):
        super().__init__(directed, weighted)

    def add_node(self, value):
        try:
            new_node = Node(value)
            self.nodes[value] = new_node
            return new_node
        except:
            print("Invalid vertex id '{}', may be exits.".format(value))

    def add_nodes(self, node_list):
        for node_name in node_list:
            new_node = Node(node_name)
            try:
                self.nodes[node_name] = new_node
            except:
                print("Invalid vertex id '{}', may be exits.".format(node_name))

    def add_edge(self, sour, terg):
        try:
            if not self.weighted:
                new_edge = Edge(False, sour, terg)
                self.edges.append(new_edge)
                self.update(new_edge)
        except:
            print("The edge already exits.")

    def add_weight_edge(self, weight, sour, terg):
        try:
            if self.weighted:
                new_edge = Edge(weight, sour, terg)
                self.edges.append(new_edge)
                self.update(new_edge)
        except:
            print("The edge already exits.")

    def add_edges(self, edges):
        """ Example to [(from, to)]
        """
        for edge in edges:
            try:
                if not self.weighted:
                    new_edge = Edge(False, edge[0], edge[1])
                    self.edges.append(new_edge)
                    self.update(new_edge)
            except:
                print("The edge already exits.")

    def update(self, edge):
        self.nodes.get(edge.head).indegree += 1
        self.nodes.get(edge.tail).outdegree += 1
        self.nodes.get(edge.tail).nexts.append(edge.head)
        self.nodes.get(edge.tail).edges.append(edge)

    def subgraph(self, nodes):
        """ Here nodes is list about node's value.
        """
        new_graph = Graph()
        for node_key in nodes:
            node = self.nodes.get(node_key)
            neighbors = list(set(node.nexts).intersection(set(nodes)))
            node.nexts = neighbors
            new_graph.nodes[node_key] = node
        return new_graph

    def jaccard_coefficient(self, node1, node2):
        """ This coefficient use to measure the ratio between the number of two node's
            common neighbors and them total neighbors.
        """
        first_node = self.nodes.get(node1).nexts
        second_node = self.nodes.get(node2).nexts
        common_friends = list(set(first_node).intersection(set(second_node)))
        return common_friends.__len__() / (first_node.__len__() + second_node.__len__())

    def total_neighbors(self,node1, node2):
        """ If node1 and node2 are exits, them total neighbors are the add between the
            node1 and node2.
        """
        first_node = self.nodes.get(node1).nexts
        second_node = self.nodes.get(node2).nexts
        return first_node.__len__() + second_node.__len__()

    def preference_attachment(self, node1, node2):
        """ the probability that two users yield an association in certain period of
            time is proportional to the product of the number of one user's neighborhoods
            and that of another's neighborhoods.
        """
        first_node = self.nodes.get(node1).nexts
        second_node = self.nodes.get(node2).nexts
        return first_node.__len__() * second_node.__len__()

    def friend_measure(self, node1, node2):
        """ the more connections their neighborhoods have with each other, the higher the
            chances the two users are connected in a social network.
        """
        first_node = self.nodes.get(node1).nexts
        second_node = self.nodes.get(node2).nexts
        F_fm = 0
        for node_key in first_node:
            x_node = self.nodes.get(node_key)
            for node_key_sec in second_node:
                #y_node = self.nodes.get(node_key_sec)
                if node_key is not node_key_sec and node_key_sec not in x_node.nexts:
                    F_fm +=1
        return F_fm


def SCC_graph(graph):
    """ calculate the number of strongly connected components within neighborhood-subgraph.
        First generate 1-dimensional neighborhood-subgraph with source and target node.

        a sample of graph, use dict to store graph.
        G = {
            'a': {'b', 'c'},
            'b': {'d', 'e', 'i'},
            'c': {'d'},
            'd': {'a', 'h'},
            'e': {'f'},
            'f': {'g'},
            'g': {'e', 'h'},
            'h': {'i'},
            'i': {'h'}
        }
    """
    nodes = graph.nodes

    # reverse graph
    def reverse_graph(nodes):
        re_graph = dict()

        for key in nodes.keys():
            re_graph[key] = nodes.get(key, set())
        # reverse edge
        for key in nodes.keys():
            for nei in nodes[key].nexts:
                re_graph[nei].nexts.append(key)
        return re_graph

    # gain a sequence sorted by time
    def topo_sort(nodes):
        res = []
        S = set()

        # Depth-first traversal/search
        def dfs(nodes, u):
            if u in S:
                return
            S.add(u)
            for v in nodes[u].nexts:
                if v in S:
                    continue
                dfs(nodes, v)
            res.append(u)

        # check each node was leave out
        for u in nodes.keys():
            dfs(nodes, u)

        res.reverse()
        return res

    # gain singe strongly connected components with assigns start node
    def walk(nodes, s, S=None):
        if S is None:
            s = set()
        Q = []
        P = dict()
        Q.append(s)
        P[s] = None
        while Q:
            u = Q.pop()
            for v in nodes[u].nexts:
                if v in P.keys() or v in S:
                    continue
                Q.append(v)
                P[v] = P.get(v, u)
        return P

    # use to record the node of strongly connected components
    seen = set()
    # to store scc
    scc = []
    GT = reverse_graph(nodes)
    for u in topo_sort(nodes):
        if u in seen:
            continue
        C = walk(GT, u, seen)
        seen.update(C)
        scc.append(sorted(list(C.keys())))

    print(scc)
