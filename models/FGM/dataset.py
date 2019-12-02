#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 15:59
# @Author  : ChenSir
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
import re


def GetLabels(data_file):
    data = open(data_file, "r")
    labels = []
    pattern = "[/?/+]([01])"
    for line in data:
        result = re.match(pattern, line)
        if result:
            labels.append(int(result.group(1)))

    return labels


class Enum:
    KNOWN_LABEL = 0
    UNKNOWN_LABEL = 1


class Config:
    # train_file = "../../dataset/loc-data/Gowalla_data_train.txt"
    # test_file = "../../dataset/loc-data/Gowalla_data_test.txt"
    # pred_file = "../../dataset/loc-data/pred.txt"
    train_file = "../../dataset/loc-data/epinions.dat"
    test_file = "../../dataset/loc-data/epinions.dat"
    pred_file = "../../dataset/loc-data/pred.txt"
    dict_file = "../../dataset/loc-data/"
    src_model_file = "../../dataset/loc-data/model/src_model.txt"
    dst_model_file = "../../dataset/loc-data/model/dst_model.txt"
    eps = 1e-3
    max_iter = 50
    max_bp_iter = 50
    gradient_step = 0.1
    has_attrib_value = True
    optimization_method = 1 # GradientDescend
    eval_each_iter = True
    penalty_sigma_square = 0.0001


class DataNode:
    def __init__(self, label_type, label, num_attrib):
        self.label_type = label_type
        self.label = label
        self.num_attrib = num_attrib
        self.attrib = []
        self.value = []


class DataEdge:
    def __init__(self, v, u, edge_type=0):
        self.v, self.u, self.edge_type = v, u, edge_type


class DataTriangle:
    def __init__(self, v, u, z, edge_type=0):
        self.v, self.u, self.z, self.edge_type = v, u, z, edge_type


class DataSample:
    def __init__(self, node, edge, triangle):
        self.num_node, self.num_edge, self.num_triangle = 0, 0, 0
        self.node = node
        self.edge = edge
        self.triangle = triangle
        self.color = {}
        if self.node:
            self.num_node = self.node.__len__()
        if self.edge:
            self.num_node = self.edge.__len__()
        if self.triangle:
            self.num_triangle = self.triangle.__len__()


class DataSet:
    def __init__(self, data_file, config):
        self.num_label, self.num_sample, self.num_attrib_type, self.num_edge_type = 0, 1, 0, 0
        self.sample = DataSample
        self.attrib_name = set()
        self.edge_type_name = set()
        self.data_file = data_file
        self.config = config
        self.labels = GetLabels(self.data_file)
        self.LoadData()

    def LoadData(self):
        data = open(self.data_file)
        nodes, edges, triangles = [], [], []
        for line in data:
            cells = line.replace("\n", "").split(" ")
            if cells[0] == "#edge":
                edge = DataEdge(cells[1], cells[2])
                edges.append(edge)
                self.edge_type_name.add(cells[3])
            elif cells[0] == "#triangle":
                triangle = DataTriangle(cells[1],cells[2],cells[3])
                triangles.append(triangle)
            else:
                label_type = cells[0][0]
                if label_type is "+":
                    num_label_type = Enum.KNOWN_LABEL
                elif label_type is "?":
                    num_label_type = Enum.UNKNOWN_LABEL
                else:
                    print("Data format wrong! Label must start with +/? DATA: {}".format(cells))
                    break
                node = DataNode(num_label_type, cells[0][1], 0)
                for attr in iter(cells[1:]):
                    attrs = attr.split(":")
                    if self.config.has_attrib_value:
                        self.attrib_name.add(attrs[0])
                        node.attrib.append(attrs[0])
                        node.value.append(attrs[1])
                    else:
                        self.attrib_name.add(attrs[0])
                        node.attrib.append(attrs[0])
                        node.value.append(1)
                node.num_attrib = cells.__len__() - 1
                nodes.append(node)
        self.sample = DataSample(nodes, edges, triangles)
        self.num_label = set(self.labels).__len__()
        self.num_attrib_type = self.attrib_name.__len__()
        self.num_edge_type = self.edge_type_name.__len__()
        if self.num_edge_type is 0:
            self.num_edge_type = 1
        data.close()


class GlobalDataSet:
    pass
