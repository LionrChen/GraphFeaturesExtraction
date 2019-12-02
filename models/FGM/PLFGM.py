#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 13:59
# @Author  : ChenSir
# @File    : PLFGM.py
# @Software: PyCharm

from __future__ import print_function

from functools import reduce

from models.FGM.graph import Graph
from models.FGM.node import Enum
import numpy as np
from math import exp, sqrt, log


class EdgeFactorFunction(object):
    def __init__(self, num_label, p_lambda, *args):
        self.num_label = num_label
        self.p_lambda = p_lambda
        self.feature_offset = args

    def CalProduct(self, y1, y2):
        i = self.feature_offset[y1 * self.num_label + y2]
        return exp(i)

    def CalProductTD(self, y1, y2, y3):
        i = self.feature_offset[y1 ** self.num_label + y2 * self.num_label + y3]
        return exp(i)


class Config(object):
    pass


def same_type_label(num_label, y, label):
    if num_label is not 6:
        return True
    return (y < 4) is (label < 4)


class FactorGraphModel:
    def __init__(self, config, train_data, test_data):
        self.triangle_func_list = []
        self.edge_func_list = []
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.num_sample = train_data.num_sample
        self.num_label = train_data.num_label
        self.num_attrib_type = train_data.num_attrib_type
        self.num_edge_type = train_data.num_edge_type
        self.num_triangle_type = 3

        self.num_edge_feature_each_type = 0
        self.num_attrib_parameter = 0
        self.num_triangle_feature = []

        self.num_feature = 0
        self.features = []

        self.parameters = [0.0 for _ in range(self.num_feature)]

        self.edge_feature_offset = []
        self.triangle_feature_offset = [[]]
        self.GenFeature()
        self.factor_graph = Graph

        self.SetupFactorGraphs()

    def GenFeature(self):
        num_attrib_parameter = self.num_label * self.num_attrib_type
        self.num_feature += num_attrib_parameter
        self.edge_feature_offset.clear()
        offset = 0
        for y1 in range(self.num_label):
            for y2 in range(y1, self.num_label):
                self.edge_feature_offset.insert(y1 * self.num_label + y2, offset)
                self.edge_feature_offset.insert(y2 * self.num_label + y1, offset)
                offset += 1
        self.num_edge_feature_each_type = offset
        self.num_feature += self.num_edge_type * self.num_edge_feature_each_type

        self.triangle_feature_offset.clear()
        print()
        offset = 0
        for i in range(3):
            self.triangle_feature_offset.append([0 for _ in range(self.num_label ** 2 + self.num_label + 1)])
        for y1 in range(self.num_label):
            for y2 in range(y1, self.num_label):
                for y3 in range(y2, self.num_label):
                    self.triangle_feature_offset[0].insert(y1 * (self.num_label ** 2) + y2 * self.num_label + y3, offset)
                    self.triangle_feature_offset[0].insert(y1 * (self.num_label ** 2) + y3 * self.num_label + y2, offset)
                    self.triangle_feature_offset[0].insert(y2 * (self.num_label ** 2) + y1 * self.num_label + y3, offset)
                    self.triangle_feature_offset[0].insert(y2 * (self.num_label ** 2) + y3 * self.num_label + y1, offset)
                    self.triangle_feature_offset[0].insert(y3 * (self.num_label ** 2) + y1 * self.num_label + y2, offset)
                    self.triangle_feature_offset[0].insert(y3 * (self.num_label ** 2) + y2 * self.num_label + y1, offset)
                    offset += 1
        self.num_triangle_feature.append(offset)
        offset = 0
        for y1 in range(self.num_label):
            for y2 in range(y1, self.num_label):
                for y3 in range(y2, self.num_label):
                    self.triangle_feature_offset[1][y1 ** self.num_label + y2 * self.num_label + y3] = offset
                    offset += 1
        self.num_triangle_feature.append(offset)
        offset = 0
        for y1 in range(self.num_label):
            for y2 in range(y1, self.num_label):
                for y3 in range(y2, self.num_label):
                    self.triangle_feature_offset[2].insert(y1 * (self.num_label ** 2) + y2 * self.num_label + y3,
                                                           offset)
                    self.triangle_feature_offset[2].insert(y1 * (self.num_label ** 2) + y3 * self.num_label + y2,
                                                           offset)
                    self.triangle_feature_offset[2].insert(y2 * (self.num_label ** 2) + y1 * self.num_label + y3,
                                                           offset)
                    self.triangle_feature_offset[2].insert(y2 * (self.num_label ** 2) + y3 * self.num_label + y1,
                                                           offset)
                    self.triangle_feature_offset[2].insert(y3 * (self.num_label ** 2) + y1 * self.num_label + y2,
                                                           offset)
                    self.triangle_feature_offset[2].insert(y3 * (self.num_label ** 2) + y2 * self.num_label + y1,
                                                           offset)
                    offset += 1
        self.num_triangle_feature.append(offset)
        for i in range(self.num_triangle_type):
            self.num_feature += self.num_triangle_feature[i]

    def SetupFactorGraphs(self):
        point_parameter = self.num_attrib_parameter
        for i in range(self.num_edge_type):
            self.edge_func_list.append(EdgeFactorFunction(self.num_label, point_parameter, self.edge_feature_offset))
            point_parameter += self.num_edge_feature_each_type

        for i in range(self.num_triangle_type):
            self.triangle_func_list.append(EdgeFactorFunction(self.num_label, point_parameter, self.triangle_feature_offset))
            point_parameter += self.num_triangle_feature[i]

        for s in range(self.num_sample):
            sample = self.train_data.sample

            self.factor_graph = Graph(sample.num_node, sample.num_edge + sample.num_triangle, self.num_label)
            for i in range(sample.num_node):
                self.factor_graph.setVariableLabel(i, sample.node[i].label)
                self.factor_graph.var[i].label_type = sample.node[i].label_type
            for i in range(sample.num_edge):
                self.factor_graph.addEdge(sample.edge[i].a, sample.edge[i].b, self.edge_func_list[sample.node[i].edge_type])
            for i in range(sample.num_triangle):
                self.factor_graph.addTriangle(sample.triangle[i].a, sample.triangle[i].b, sample.triangle[i].c,
                                              self.triangle_func_list[sample.triangle[i].triangle_type])

    def Train(self):
        gradient = []
        old_f = 0.0
        eps = self.config.eps
        iflag = 1

        num_iter = 0
        while iflag is not 0 and num_iter < self.config.max_iter:

            f = self.CalcGradient(gradient)

            print("[Iter {}] log-likelihood: {}\n".format(num_iter, f))

            if abs(old_f - f) < eps:
                break

            old_f = f
            # Negate f and gradient vector because the LBFGS optimization below minimizes the ojective function
            # while we would like to maximize it
            f *= -1
            for i in range(self.num_feature):
                gradient[i] *= -1

            # Invoke L-BFGS

            if self.config.optimization_method is 0:
                if iflag < 0:
                    break
            else:
                g_norm = 0.0
                for i in range(self.num_feature):
                    g_norm += gradient[i] * gradient[i]
                g_norm = sqrt(g_norm)

                if g_norm > 1e-8:
                    for i in range(self.num_feature):
                        gradient[i] /= g_norm

                for i in range(self.num_feature):
                    self.parameters[i] -= gradient[i] * self.config.gradient_step

                iflag = 1

            num_iter += 1

    def CalcGradient(self, gradient):
        f = 0.0
        for i in range(self.num_feature):
            gradient.append(0)
        return self.CalcPartialLabeledGradientForSample(self.train_data.sample, self.factor_graph, gradient)

    def CalcGradientForSample(self, sample, factor_graph, gradient):
        factor_graph.clearDataForSumProduct()

        # set state_factor
        p_lambda = 0
        for i in range(sample.num_node):
            state_factor = []
            for y in range(self.num_label):
                if sample.node[i].label_type is Enum.KNOWN_LABEL and y is not sample.node[i].label or same_type_label(
                        self.num_label, y, sample.node[i].label):
                    state_factor.append(0)
                else:
                    lambda_vec = self.parameters[p_lambda:sample.node[i].num_attrib]
                    value_vec = sample.node[i].value
                    state_factor.append(exp(reduce(np.multiply, np.multiply(lambda_vec, value_vec))))
                p_lambda += self.num_attrib_type
            factor_graph.setVariableStateFactor(i, state_factor)

        factor_graph.beliefPropagation(self.config.max_bp_iter)
        factor_graph.marginals()

        # Calculate gradient & log-likelihood
        f, Z = 0.0, 0.0
        for i in range(sample.num_node):
            y = sample.node[i].label
            for t in range(sample.node[i].num_attrib):
                f += self.parameters[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] * sample.node[i].value[
                    t]
        for i in range(sample.num_edge):
            a = sample.node[sample.edge[i].a].label
            b = sample.node[sample.edge[i].b].label
            f += self.parameters[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)]

        # calc log-likelihood
        # using Bethe Approximation
        for i in range(sample.num_node):
            for y in range(self.num_label):
                for t in range(sample.node[i].num_attrib):
                    Z += self.parameters[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] * \
                         sample.node[i].value[t] * factor_graph.var[i].marginal[y]
        for i in range(factor_graph.num_fac_node):
            for a in range(self.num_label):
                for b in range(self.num_label):
                    Z += self.parameters[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)] * \
                         factor_graph.fac[i].marginal[a][b]

        # edge entropy
        for i in range(factor_graph.num_fac_node):
            h_e = 0.0
            for a in range(self.num_label):
                for b in range(self.num_label):
                    if factor_graph.fac[i].marginal[a][b] > 1e-10:
                        h_e += -factor_graph.fac[i].marginal[a][b] * log(factor_graph.fac[i].marginal[a][b])

            Z += h_e
        # node entropy
        for i in range(sample.num_node):
            h_v = 0.0
            for y in range(self.num_label):
                if abs(factor_graph.var[i].marginal[y]) > 1e-10:
                    h_v += - factor_graph.var[i].marginal[y] * log(factor_graph.var[i].marginal[y])
            Z -= h_v * len(factor_graph.var[i].nbrs) - 1
        f -= Z

        # calc gradient part: -E_{Y} f_i
        for i in range(sample.num_node):
            for t in range(sample.node[i].num_attrib):
                gradient[self.GetAttribParameterIndex(sample.node[i].label, sample.node[i].attrib[t])] += \
                    sample.node[i].value[t]
        for i in range(factor_graph.num_fac_node):
            gradient[self.GetEdgeParameterIndex(sample.edge[i].edge_type,
                                                sample.node[sample.edge[i].u].label,
                                                sample.node[sample.edge[i].v].label)] += 1.0
        # expectation
        for i in range(sample.num_node):
            for y in range(self.num_label):
                for t in range(sample.node[i].num_attrib):
                    gradient[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] -= \
                        sample.node[i].value[t] * factor_graph.var[i].marginal[y]
        for i in range(factor_graph.num_fac_node):
            for a in range(self.num_label):
                for b in range(self.num_label):
                    gradient[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)] -= \
                        factor_graph.var[i].marginal[a][b]
        return f

    def CalcPartialLabeledGradientForSample(self, sample, factor_graph, gradient):
        factor_graph.labeled_given = True
        factor_graph.clearDataForSumProduct()

        # set state_factor
        p_lambda = 0
        for i in range(sample.num_node):
            state_factor = []
            for y in range(self.num_label):
                if sample.node[i].label_type is Enum.KNOWN_LABEL and y is not sample.node[i].label or same_type_label(
                        self.num_label, y, sample.node[i].label):
                    state_factor.append(0)
                else:
                    lambda_vec = self.parameters[p_lambda:sample.node[i].num_attrib]
                    value_vec = sample.node[i].value
                    state_factor.append(exp(reduce(np.multiply, np.multiply(lambda_vec, value_vec))))
                p_lambda += self.num_attrib_type
            factor_graph.setVariableStateFactor(i, state_factor)

        factor_graph.beliefPropagation(self.config.max_bp_iter)
        factor_graph.marginals()

        """ Gradient = E_{Y|Y_L} f_i - E_{Y} f_i
        
            calc gradient part : + E_{Y|Y_L} f_i
            """
        for i in range(sample.num_node):
            for y in range(self.num_label):
                for t in range(sample.node[i].num_attrib):
                    gradient[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] += \
                        sample.node[i].value[t] * factor_graph.var[i].marginal[y]

        for i in range(factor_graph.num_fac_node):
            if len(factor_graph.fac[i].nbrs) is 2:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        gradient[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)] += \
                            factor_graph.fac[i].marginal[a][b]
            else:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        for c in range(self.num_label):
                            gradient[
                                self.GetTriangleParameterIndex(sample.triangle[i - sample.num_edge].triangle_type, a, b,
                                                               c)] += \
                                factor_graph.fac[i].marginalTD[a][b][c]
        factor_graph.beliefPropagation(self.config.max_bp_iter)
        factor_graph.marginals()

        # calc gradient part: -E_{Y} f_i
        for i in range(sample.num_node):
            for y in range(self.num_label):
                for t in range(sample.node[i].num_attrib):
                    gradient[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] -= \
                        sample.node[i].value[t] * factor_graph.var[i].marginal[y]
        for i in range(factor_graph.num_fac_node):
            if len(factor_graph.fac[i].nbrs) is 2:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        gradient[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)] -= \
                            factor_graph.fac[i].marginal[a][b]
            else:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        for c in range(self.num_label):
                            gradient[
                                self.GetTriangleParameterIndex(sample.triangle[i - sample.num_edge].triangle_type, a, b,
                                                               c)] -= \
                                factor_graph.fac[i].marginalTD[a][b][c]
        f, Z = 0.0, 0.0
        for i in range(sample.num_node):
            y = sample.node[i].label
            for t in range(sample.node[i].num_attrib):
                f += self.parameters[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] * sample.node[i].value[
                    t]
        for i in range(sample.num_edge):
            a = sample.node[sample.edge[i].u].label
            b = sample.node[sample.edge[i].v].label
            f += self.parameters[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)]
        for i in range(sample.num_triangle):
            a = sample.node[sample.triangle[i].u].label
            b = sample.node[sample.triangle[i].v].label
            c = sample.node[sample.triangle[i].z].label
            f += self.parameters[self.GetTriangleParameterIndex(sample.triangle[i].triangle_type, a, b, c)]

        # calc log-likelihood
        # using Bethe Approximation
        for i in range(sample.num_node):
            for y in range(self.num_label):
                for t in range(sample.node[i].num_attrib):
                    Z += self.parameters[self.GetAttribParameterIndex(y, sample.node[i].attrib[t])] * \
                         sample.node[i].value[t] * factor_graph.var[i].marginal[y]
        for i in range(factor_graph.num_fac_node):
            if len(factor_graph.fac[i].nbrs) is 2:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        Z += self.parameters[self.GetEdgeParameterIndex(sample.edge[i].edge_type, a, b)] * \
                             factor_graph.fac[i].marginal[a][b]
            else:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        for c in range(self.num_label):
                            Z += self.parameters[
                                     self.GetTriangleParameterIndex(sample.triangle[i - sample.num_edge].triangle_type,
                                                                    a, b,
                                                                    c)] * \
                                 factor_graph.fac[i].marginalTD[a][b][c]

        # edge entropy
        for i in range(factor_graph.num_fac_node):
            h_e = 0.0
            if len(factor_graph.fac[i].nbrs) is 2:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        if factor_graph.fac[i].marginal[a][b] > 1e-10:
                            h_e += -factor_graph.fac[i].marginal[a][b] * log(factor_graph.fac[i].marginal[a][b])
            else:
                for a in range(self.num_label):
                    for b in range(self.num_label):
                        for c in range(self.num_label):
                            if factor_graph.fac[i].marginal[a][b][c] > 1e-10:
                                h_e += -factor_graph.fac[i].marginal[a][b][c] * log(
                                    factor_graph.fac[i].marginal[a][b][c])
            Z += h_e
        # edge entropy
        for i in range(sample.num_node):
            h_v = 0.0
            for y in range(self.num_label):
                if abs(factor_graph.var[i].marginal[y]) > 1e-10:
                    h_v += - factor_graph.var[i].marginal[y] * log(factor_graph.var[i].marginal[y])
            Z -= h_v * len(factor_graph.var[i].nbrs) - 1
        f -= Z

        # Let's take a look of current accuracy
        factor_graph.clearDataForSumProduct()
        factor_graph.labeled_given = True
        for i in range(sample.num_node):
            p_lambda = 0
            state_factor = []
            for y in range(self.num_label):
                lambda_vec = self.parameters[p_lambda:sample.node[i].num_attrib]
                value_vec = sample.node[i].value
                state_factor.append(exp(reduce(np.multiply, np.multiply(lambda_vec, value_vec))))
                p_lambda += self.num_attrib_type
            factor_graph.setVariableStateFactor(i, state_factor)

        factor_graph.maxSumPropagation(self.config.max_bp_iter)

        inf_label = []
        for i in range(sample.num_node):
            ybest, vbest = -1, 0
            for y in range(self.num_label):
                v = factor_graph.var[i].state_fac[y]
                for nb in range(len(factor_graph.var[i].nbrs)):
                    v *= factor_graph.var[i].belief[nb][y]
                if same_type_label(self.num_label, y, sample.node[i].label) and (ybest < 0 or v > vbest):
                    ybest, vbest = y, v
            inf_label.append(ybest)

        hit, miss, hitu, missu = 0, 0, 0, 0
        cnt, ucnt = np.zeros([50, 50]), np.zeros([50, 50])

        for i in range(sample.num_node):
            if inf_label[i] is sample.node[i].label:
                hit += 1
            else:
                miss += 1
            cnt[inf_label[i]][sample.node[i].label] += 1
            if sample.node[i].label_type is Enum.UNKNOWN_LABEL:
                if inf_label[i] is sample.node[i].label:
                    hitu += 1
                else:
                    missu += 1
            ucnt[inf_label[i]][sample.node[i].label] += 1
        print("A_HIT = {}, U_HIT = {}".format(hit, hitu, ".4d"))
        print("A_MISS = {}, U_MISS = {}".format(miss, missu, ".4d"))

        print("A_Accuracy = {}".format(hit / (hit + miss), ".4f"))

        for i in range(self.num_label):
            srow, scolumn = 0, 0
            for j in range(self.num_label):
                srow += cnt[i][j]
                scolumn += cnt[j][i]
            a_precision = cnt[i][i] / srow
            a_recall = cnt[i][i] / scolumn
            a_fmeasure = 2.0 * a_precision * a_recall / (a_precision + a_recall)
            print("A_Precision = {}, A_Recall = {:.4f}, A_Fmeasure = {:.4f}".format(a_precision, a_recall, a_fmeasure))

        return f

    def SelfEvaluate(self):
        sample = self.train_data.sample
        n = sample.num_node
        m = sample.num_edge
        factor_graph = Graph(n, m, self.num_label)
        for i in range(m):
            factor_graph.addEdge(sample.edge[i].u, sample.edge[i].v, self.edge_func_list[sample.edge[i].edge_type])
        factor_graph.genPropagateOrder()
        factor_graph.clearDataForSumProduct()
        for i in range(sample.num_node):
            p_lambda = 0
            state_factor = []
            for y in range(self.num_label):
                lambda_vec = self.parameters[p_lambda:sample.node[i].num_attrib]
                value_vec = sample.node[i].value
                state_factor.append(exp(reduce(np.multiply, np.multiply(lambda_vec, value_vec))))
                p_lambda += self.num_attrib_type
            factor_graph.setVariableStateFactor(i, state_factor)

        factor_graph.maxSumPropagation(self.config.max_bp_iter)

        inf_label = []
        for i in range(sample.num_node):
            ybest, vbest = -1, 0
            for y in range(self.num_label):
                v = factor_graph.var[i].state_fac[y]
                for nb in range(len(factor_graph.var[i].nbrs)):
                    v *= factor_graph.var[i].belief[nb][y]
                if same_type_label(self.num_label, y, sample.node[i].label) and (ybest < 0 or v > vbest):
                    ybest, vbest = y, v
            inf_label.append(ybest)

        curt_tot, curt_hit = 0, 0
        for i in range(n):
            curt_tot += 1
            if inf_label[i] is sample.node[i].label:
                curt_hit += 1

        print("Accuracy {:.4f} / {:.4f} : {:.6f}\n".format(curt_hit, curt_tot, curt_hit/curt_tot))

    def InitEvaluate(self, config, test_data):
        self.config = config
        self.test_data = test_data

    def Evaluate(self):
        pred_file = open(self.config.pred_file, "w")
        sample = self.test_data.sample
        n = sample.num_node
        m = sample.num_edge
        o = sample.num_triangle
        factor_graph = Graph(n, m + o, self.num_label)
        for i in range(m):
            factor_graph.addEdge(sample.edge[i].u, sample.edge[i].v, self.edge_func_list[sample.edge[i].edge_type])
        for i in range(o):
            factor_graph.addTriangle(sample.triangle[i].u, sample.triangle[i].v, sample.triangle[i].z,
                                     self.triangle_func_list[sample.triangle[i].triangle_type])
        factor_graph.genPropagateOrder()
        factor_graph.clearDataForSumProduct()
        for i in range(sample.num_node):
            p_lambda = 0
            state_factor = []
            for y in range(self.num_label):
                lambda_vec = self.parameters[p_lambda:sample.node[i].num_attrib]
                value_vec = sample.node[i].value
                state_factor.append(exp(reduce(np.multiply, np.multiply(lambda_vec, value_vec))))
                p_lambda += self.num_attrib_type
            factor_graph.setVariableStateFactor(i, state_factor)

        factor_graph.maxSumPropagation(self.config.max_bp_iter)

        inf_label = []
        for i in range(sample.num_node):
            ybest, vbest = -1, 0
            for y in range(self.num_label):
                v = factor_graph.var[i].state_fac[y]
                for nb in range(len(factor_graph.var[i].nbrs)):
                    v *= factor_graph.var[i].belief[nb][y]
                if same_type_label(self.num_label, y, sample.node[i].label) and (ybest < 0 or v > vbest):
                    ybest, vbest = y, v
            inf_label.append(ybest)

        curt_tot, curt_hit = 0, 0
        cnt = np.zeros((50,50))
        for i in range(n):
            curt_tot += 1
            if inf_label[i] is sample.node[i].label:
                curt_hit += 1
            cnt[inf_label[i]][sample.node[i].label] += 1
        print("Accuracy {:.4f} / {:.4f} : {:.6f}\n".format(curt_hit, curt_tot, curt_hit / curt_tot))

        for i in range(self.num_label):
            srow, scolumn = 0, 0
            for j in range(self.num_label):
                srow += cnt[i][j]
                scolumn += cnt[j][i]
            a_precision = cnt[i][i] / srow
            a_recall = cnt[i][i] / scolumn
            a_fmeasure = 2.0 * a_precision * a_recall / (a_precision + a_recall)
            print("A_Precision = {}, A_Recall = {:.4f}, A_Fmeasure = {:.4f}".format(a_precision, a_recall, a_fmeasure))
        for i in range(n):
            pred_file.writelines([inf_label[i]])
        pred_file.close()

    def SaveModel(self, file_name):
        model_file = open(file_name, "w")
        model_file.writelines(self.num_feature)
        model_file.writelines(self.parameters)
        model_file.close()

    def LoadModel(self, file_name):
        model_file = open(file_name, "r")
        self.parameters = model_file.readlines()[1:]
        model_file.close()

    def GetAttribParameterIndex(self, y, x):
        return y * self.num_attrib_type + x

    def GetEdgeParameterIndex(self, edge_type, a, b):
        offset = self.edge_feature_offset[a * self.num_label + b]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    def GetTriangleParameterIndex(self, triangle_type, a, b, c):
        offset = self.triangle_feature_offset[triangle_type][
            a * self.num_label * self.num_label + b * self.num_label + c]
        id = self.num_attrib_parameter + self.num_edge_type * self.num_edge_feature_each_type + offset
        for i in range(triangle_type):
            id += self.num_triangle_feature[i]
        return id
