#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 19:53
# @Author  : ChenSir
# @File    : RandWalk.py
# @Software: PyCharm

import networkx as nx
import random
import csv


class RandomWalkSample:
    def __init__(self, scale, graph=None):
        self.graph = graph
        self.scale = scale
        self.num_node = 0
        self.num_edges = 0
        self.start_nodes = []
        self.preview_node = None
        self.current_node = None
        self.prob_back = 15
        self.num_restart = 100
        self.distance = 0
        self.sample_graph = nx.Graph()

    def build_graph(self, edges_df):
        vertices = list(set((edges_df.iloc[:, 1]).tolist() + (edges_df.iloc[:, 2]).tolist()))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(vertices)
        for index, row in edges_df.iterrows():
            self.graph.add_edge(row[1], row[2], weight=row[0])
        self.num_node = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        print("Graph build success. Almost {} edges was processed.".format(self.num_edges))
        return self.graph

    def reset_attr(self):
        self.preview_node = None
        self.current_node = None
        self.distance = 0

    def get_random_node(self):
        rand_node = random.choice(list(self.graph.nodes()))
        return rand_node

    def get_start_node(self):
        start_node = self.get_random_node()
        if start_node not in self.start_nodes:
            self.start_nodes.append(start_node)
            self.preview_node = start_node
            self.current_node = start_node
            return start_node
        else:
            return self.get_start_node()

    def to_walk(self, edge, weight):
        self.sample_graph.add_node(edge[1])
        self.sample_graph.add_edge(edge[0], edge[1], weight=weight)

    def walk(self, node):
        self.distance += 1
        print("Present number of sampling graph is {}.".format(self.sample_graph.number_of_nodes()))

        node_nbrs = list(self.graph.neighbors(node))
        try:
            node_nbrs.remove(self.preview_node)
        except:
            print("Preview node not in list.")
        if node_nbrs:
            walk_prob = random.randint(0, 100)
            if walk_prob > self.prob_back:
                self.sample_graph.add_node(node)
                rand_next_node = random.choice(node_nbrs)
                self.current_node = rand_next_node
                self.to_walk((node, rand_next_node), self.graph[node][rand_next_node]['weight'])
                self.preview_node = node
                return rand_next_node
            else:
                self.walk(self.preview_node)
        else:
            self.walk(self.preview_node)

    def start_rand_walk(self):
        start_node = self.get_start_node()
        start_node_nbrs = list(self.graph.neighbors(start_node))
        if start_node_nbrs:
            self.walk(start_node)
        else:
            self.start_rand_walk()
        while self.sample_graph.number_of_nodes() < self.scale:
            if self.distance < self.num_restart:
                self.walk(self.current_node)
            else:
                self.reset_attr()
                self.start_rand_walk()

    @staticmethod
    def graph_to_csv(graph):
        output_file = open("sample_graph.csv", 'w')
        csv_write = csv.writer(output_file, dialect='excel')
        for n, nbrs in graph.adjacency():
            for nbr, eattr in nbrs.items():
                label = eattr['weight']
                csv_write.writerow([n, nbr, label])
