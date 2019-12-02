#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ChenSir'
__mtime__ = '2019/4/8'
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
from functools import reduce
from math import exp
import numpy as np

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)

    Note to self: use %pdb and %load_ext autoreload followed by %autoreload 2
"""

class Enum:
    KNOWN_LABEL = 0
    UNKNOWN_LABEL = 1


class FactorFunction:
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


class Node(object):
    """ Superclass for graph nodes
    """
    epsilon = 10 ** (-6)

    def __init__(self, nid, num_label):
        self.enabled = True
        self.nid = nid
        self.state = 1
        self.nbrs = []
        self.num_label = num_label

        self.belief = []
        self.msg = []
        self.old_msg = []
        self.diff_max = 0

    def addNeighbor(self, node):
        if node not in self.nbrs:
            self.nbrs.append(node)

    def GetMessageFrom(self, f, msg, diff_max):
        i = self.nbrs.index(f)
        self.old_msg[i] = self.belief[i]
        self.belief[i] = msg
        arr = np.append(np.array(self.belief[i]) - np.array(msg), diff_max)
        self.diff_max = max(arr)
        return max(arr)

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
        for n in self.nbrs:
            # don't call enable() as it will recursively enable entire graph
            n.enabled = True

    def NormalizeMessages(self):
        """ Normalize to sum to 1
        """
        self.msg = [x / np.sum(x) for x in self.msg]

    def receiveMessage(self, f, m):
        """ Places new message into correct location in new message list
        """
        if self.enabled:
            i = self.nbrs.index(f)
            self.belief[i] = m

    def sendMessages(self):
        """ Sends all outgoing messages
        """
        for i in range(0, len(self.msg)):
            self.nbrs[i].receiveMessage(self, self.msg[i])

    def checkConvergence(self):
        """ Check if any messages have changed
        """
        if self.enabled:
            for i in range(0, len(self.msg)):
                # check messages have same shape
                self.old_msg[i].shape = self.old_msg[i].shape
                delta = np.absolute(self.msg[i] - self.old_msg[i])
                if (delta > Node.epsilon).any():  # if there has been change
                    return False
            return True
        else:
            # Always return True if disabled to avoid interrupting check
            return True


class VarNode(Node):
    """ Variable node in factor graph
    """

    def __init__(self, nid, num_label):
        super(VarNode, self).__init__(nid, num_label)
        self.observed = -1  # only >= 0 if variable is observed
        self.state_fec = [0.0 for _ in range(num_label)]
        self.y = 1
        self.label_type = 0
        self.marginal = [0.0 for _ in range(num_label)]

    def BeliefPropagation(self, diff_max, labeled_given):
        self.prepMessages()
        for i in range(0, len(self.msg)):
            return self.nbrs[i].GetMessageFrom(self, self.msg[i], self.diff_max)

    def MaxSumPropagation(self, diff_max, labeled_given):
        self.prepMessages()
        for i in range(0, len(self.msg)):
            return self.nbrs[i].GetMessageFrom(self, self.msg[i], self.diff_max)

    def condition(self, observation):
        """ Condition on observing certain value
        """
        self.enable()
        self.observed = observation
        # set messages (won't change)
        for i in range(0, len(self.msg)):
            self.belief[i] = np.zeros((self.num_label, 1))
            self.belief[i][self.observed] = 1.

    def prepMessages(self):
        """ Multiplies together incoming messages to make new outgoing
        """

        # compute new messages if no observation has been made
        if self.enabled and self.observed < 0 and len(self.nbrs) > 1:
            # switch reference for old messages
            # self.nextStep()
            for i in range(0, len(self.belief)):
                # multiply together all excluding message at current index
                curr = self.belief[:]
                del curr[i]
                self.msg[i] = reduce(np.multiply, curr, np.array(self.state_fec))

            # normalize once finished with all messages
            self.NormalizeMessages()


class FacNode(Node):
    """ Factor node in factor graph
    """

    def __init__(self, nid, num_label):
        super(FacNode, self).__init__(nid, num_label)
        self.state_factor_arr = [self.state for x in self.num_label]
        self.factor_func = FactorFunction()

        self.marginal = np.zeros((self.num_label, self.num_label), dtype=float)
        self.marginalTD = np.zeros((self.num_label, self.num_label, self.num_label), dtype=float)
        # num of edges
        numNbrs = len(self.nbrs)

        # init messages
        for i in range(0, numNbrs):
            v = self.nbrs[i]
            vdim = v.num_label

            # init for factor
            self.belief.append(np.ones((vdim, 1)))

            # init for variable
            v.nbrs.append(self)
            v.belief.append(np.ones((vdim, 1)))

        # error check
        # assert (numNbrs == numDependencies), "Factor dimensions does not match size of domain."

    def setFactorFunction(self, func):
        self.factor_func = func

    def reset(self):
        super(FacNode, self).reset()
        for i in range(0, len(self.belief)):
            self.belief[i] = np.ones((self.nbrs[i].dim, 1))

    def BeliefPropagation(self, diff_max, labeled_given):
        self.prepMessages(labeled_given)
        for i in range(0, len(self.msg)):
            return self.nbrs[i].GetMessageFrom(self, self.msg[i], self.diff_max)

    def MaxSumPropagation(self, diff_max, labeled_given):
        self.prepMaxSumMessages(labeled_given)
        for i in range(0, len(self.msg)):
            return self.nbrs[i].GetMessageFrom(self, self.msg[i], self.diff_max)

    def prepMessages(self, labeled_given):
        """ Multiplies incoming messages w/ P to make new outgoing
        """
        mnum = len(self.nbrs)

        if mnum is 2:
            for i in range(0, 2):
                if labeled_given and self.nbrs[i].label_type is Enum.KNOWN_LABEL:
                    self.msg[i] = np.ones((self.num_label, 0))
                    self.msg[i][self.nbrs[i].y] = 1.0
                else:
                    for i in range(0, len(self.belief)):
                        # multiply together all excluding message at current index
                        curr = self.belief[:]
                        del curr[i]
                        h_vec = [self.factor_func.CalProduct(i, y) for y in range(0, self.num_label)]
                        self.msg[i] = reduce(np.multiply, np.append(curr, [h_vec], axis=0))
                    self.NormalizeMessages()
        else:
            for i in range(0, 3):
                if labeled_given and self.nbrs[i].label_type is Enum.KNOWN_LABEL:
                    self.msg[i] = np.ones((self.num_label, 0))
                    self.msg[i][self.nbrs[i].y] = 1.0
                else:
                    curr = self.belief[:]
                    del curr[i]
                    b_vec = np.dot(np.mat(curr[0]).T, np.mat(curr[1]))
                    g_vec = []
                    for y in range(0, self.num_label):
                        g_vec_item = []
                        for y1 in range(0, self.num_label):
                            for y2 in range(0, self.num_label):
                                if i is 0:
                                    g_vec_item.append(self.factor_func.CalProductTD(y, y1, y2))
                                elif i is 1:
                                    g_vec_item.append(self.factor_func.CalProductTD(y2, y, y1))
                                else:
                                    g_vec_item.append(self.factor_func.CalProductTD(y1, y2, y))
                        g_vec.append(g_vec_item)
                    s = np.multiply(b_vec, np.array(g_vec))

                    self.msg[i] = [np.sum(s[i]) for i in range(0, self.num_label)]

                self.NormalizeMessages()

    def prepMaxSumMessages(self, labeled_given):
        """ Multiplies incoming messages w/ P to make new outgoing
        """
        mnum = len(self.nbrs)
        max_sum = -1e200

        if mnum is 2:
            for i in range(0, 2):
                if labeled_given and self.nbrs[i].label_type is Enum.KNOWN_LABEL:
                    self.msg[i] = np.ones((self.num_label, 0))
                    self.msg[i][self.nbrs[i].y] = 1.0
                else:
                    for i in range(0, len(self.belief)):
                        # multiply together all excluding message at current index
                        curr = self.belief[:]
                        del curr[i]
                        h_vec = [self.factor_func.CalProduct(i, y) for y in range(0, self.num_label)]
                        temp_arr = reduce(np.multiply, np.append(curr, [h_vec], axis=0))
                        self.msg[i] = [max_sum if item < max_sum else item for item in temp_arr]
                    self.NormalizeMessages()
        else:
            for i in range(0, 3):
                if labeled_given and self.nbrs[i].label_type is Enum.KNOWN_LABEL:
                    self.msg[i] = np.ones((self.num_label, 0))
                    self.msg[i][self.nbrs[i].y] = 1.0
                else:
                    curr = self.belief[:]
                    del curr[i]
                    b_vec = np.dot(np.mat(curr[0]).T, np.mat(curr[1]))
                    g_vec = []
                    for y in range(0, self.num_label):
                        g_vec_item = []
                        for y1 in range(0, self.num_label):
                            for y2 in range(0, self.num_label):
                                if i is 0:
                                    g_vec_item.append(self.factor_func.CalProductTD(y, y1, y2))
                                elif i is 1:
                                    g_vec_item.append(self.factor_func.CalProductTD(y2, y, y1))
                                else:
                                    g_vec_item.append(self.factor_func.CalProductTD(y1, y2, y))
                        g_vec.append(g_vec_item)
                    s = np.multiply(b_vec, np.array(g_vec))

                    temp_arr = [np.sum(s[i]) for i in range(0, self.num_label)]
                    self.msg[i] = [max_sum if item < max_sum else item for item in temp_arr]

                self.NormalizeMessages()

