#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/2 15:31
# @Author  : ChenSir
# @File    : mainFGM.py
# @Software: PyCharm

from models.FGM.dataset import Config, DataSet
from models.FGM.PLFGM import FactorGraphModel


def makeEvaluate(config, data_set, model):
    model.InitEvaluate(config, data_set)
    model.Evaluate()


def Estimate(config):
    data_set_train = DataSet(config.train_file, config)
    data_set_test = DataSet(config.test_file, config)
    print("number of label: {}".format(data_set_train.num_label))
    print("number of edge type: {}".format(data_set_train.num_edge_type))
    print("number of attrib type: {}".format(data_set_train.num_attrib_type))
    model = FactorGraphModel(config, data_set_train, data_set_test)
    model.Train()


def inference(config):
    data_set_train = DataSet(config.train_file, config)
    data_set_test = DataSet(config.test_file_file, config)
    print("number of label: {}".format(data_set_train.num_label))
    print("number of edge type: {}".format(data_set_train.num_edge_type))
    print("number of attrib type: {}".format(data_set_train.num_attrib_type))
    model = FactorGraphModel(config, data_set_train, data_set_test)
    model.LoadModel(config.src_model_file)
    makeEvaluate(config, data_set_train, model)


if __name__ == '__main__':
    config = Config()
    Estimate(config)
