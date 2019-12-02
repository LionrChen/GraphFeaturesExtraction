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
# Graph class
from __future__ import print_function
from future.utils import iteritems
from functools import reduce
import numpy as np
from models.FGM.node import Node, VarNode, FacNode

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)
"""


class Graph:
    """ Putting everything together
    """

    def __init__(self, n, m, num_label):
        self.var = {}
        self.fac = {}
        self.diff_max = 0.0
        self.converged = False
        self.labeled_given = False
        self.num_label = num_label
        self.num_ver_node = n
        self.num_fac_node = m
        self.num_node = self.num_ver_node + self.num_fac_node

        self.p_node = [Node(i, self.num_label) for i in range(0, self.num_label)]

        self.entry = []

    def setVariableLabel(self, u, y):
            self.var[u].y = y

    def setVariableStateFactor(self, u, v):
        self.var[u].state_fac = v

    def addVarNode(self, num_label):
        newId = len(self.var)
        newVar = VarNode(newId, num_label)
        self.var[newId] = newVar

        return newVar

    def addFacNode(self, num_label):
        newId = len(self.fac)
        newFac = FacNode(newId, num_label)
        self.fac[newId] = newFac

        return newFac

    def addEdge(self, a, b, func):
        fac_node = self.addFacNode(self.num_label)
        fac_node.setFactorFunction(func)
        fac_node.addNeighbor(self.var[a])
        fac_node.addNeighbor(self.var[b])
        self.var[a].addNeighbor(fac_node)
        self.var[b].addNeighbor(fac_node)

    def addTriangle(self, a, b, c, func):
        fac_node = self.addFacNode(self.num_label)
        fac_node.setFactorFunction(func)
        fac_node.addNeighbor(self.var[a])
        fac_node.addNeighbor(self.var[b])
        fac_node.addNeighbor(self.var[c])
        self.var[a].addNeighbor(fac_node)
        self.var[b].addNeighbor(fac_node)
        self.var[c].addNeighbor(fac_node)

    def clearDataForSumProduct(self):
        for var_node in self.var:
            var_node.state_fac = [1.0 / self.num_label for _ in self.num_label]
            for k in range(var_node.nbrs.__len__()):
                var_node.belief[k] = [1.0 / self.num_label for _ in self.num_label]

        for fac_node in self.fac:
            for k in range(fac_node.nbrs.__len__()):
                fac_node.belief[k] = [1.0 / self.num_label for _ in self.num_label]

    def genPropagateOrder(self):
        mark = [False for _ in range(0, self.num_node)]
        bfs_node = [Node(i, self.num_label) for i in range(0, self.num_label)]
        head = 0
        tail = -1
        for i in range(0, self.num_node):
            if not mark[i]:
                self.entry.append(self.p_node[i])
                bfs_node[++tail] = self.p_node[i]
                mark[self.p_node[i].nid] = 1
                while head <= tail:
                    head += 1
                    u = bfs_node[head]
                    for nbr in u.nbrs:
                        if not mark[nbr.id]:
                            bfs_node[++ tail] = nbr
                            mark[nbr.id] = 1

    def beliefPropagation(self, max_iter):
        """ This is the algorithm!
                    Each timestep:
                    take incoming messages and multiply together to produce outgoing for all nodes
                    then push outgoing to neighbors' incoming
                    check outgoing v. previous outgoing to check for convergence
                """
        # loop to convergence
        timestep = 0
        while timestep < max_iter and not self.converged:  # run for maxsteps cycles
            timestep = timestep + 1

            for k, f in iteritems(self.fac):
                # start with factor-to-variable
                # can send immediately since not sending to any other factors
                self.diff_max = max([f.BeliefPropagation(self.diff_max, self.labeled_given), self.diff_max])
                if self.diff_max < 1e-6:
                    break

            for k, v in iteritems(self.var):
                # variable-to-factor
                self.diff_max = max([v.BeliefPropagation(self.diff_max, self.labeled_given), self.diff_max])
                if self.diff_max < 1e-6:
                    break

        #     # check for convergence
        #     t = True
        #     for k, v in iteritems(self.var):
        #         t = t and v.checkConvergence()
        #         if not t:
        #             break
        #     if t:
        #         for f in self.fac:
        #             t = t and f.checkConvergence()
        #             if not t:
        #                 break
        #
        #     if t:  # we have convergence!
        #         self.converged = True
        #
        # # if run for 500 steps and still no convergence:impor
        # if not self.converged:
        #     print("No convergence!")

    def disableAll(self):
        """ Disable all nodes in graph
            Useful for switching on small subnetworks
            of bayesian nets
        """
        for k, v in iteritems(self.var):
            v.disable()
        for f in self.fac:
            f.disable()

    def reset(self):
        """ Reset messages to original state
        """
        for k, v in iteritems(self.var):
            v.reset()
        for f in self.fac:
            f.reset()
        self.converged = False

    def sumProduct(self, maxsteps=500):
        """ This is the algorithm!
            Each timestep:
            take incoming messages and multiply together to produce outgoing for all nodes
            then push outgoing to neighbors' incoming
            check outgoing v. previous outgoing to check for convergence
        """
        # loop to convergence
        timestep = 0
        while timestep < maxsteps and not self.converged:  # run for maxsteps cycles
            timestep = timestep + 1
            print(timestep)

            for f in self.fac:
                # start with factor-to-variable
                # can send immediately since not sending to any other factors
                f.prepMessages()
                f.sendMessages()

            for k, v in iteritems(self.var):
                # variable-to-factor
                v.prepMessages()
                v.sendMessages()

            # check for convergence
            t = True
            for k, v in iteritems(self.var):
                t = t and v.checkConvergence()
                if not t:
                    break
            if t:
                for f in self.fac:
                    t = t and f.checkConvergence()
                    if not t:
                        break

            if t:  # we have convergence!
                self.converged = True

        # if run for 500 steps and still no convergence:impor
        if not self.converged:
            print("No convergence!")

    def marginals(self):
        """ Return dictionary of all marginal distributions
            indexed by corresponding variable name
        """
        # for each var
        for k, v in iteritems(self.var):
            if v.enabled:  # only include enabled variables
                # multiply together messages
                vmarg = 1
                for i in range(0, len(v.belief)):
                    vmarg = vmarg * v.belief[i]

                # normalize
                n = np.sum(vmarg)
                vmarg = vmarg / n

                v.marginal = vmarg

        # for each fac
        for k, f in iteritems(self.fac):
            if f.enabled:
                if len(f.nbrs) is 2:
                    for i in range(self.num_label):
                        b_vec = np.dot(np.mat(f.belief[0]).T, np.mat(f.belief[1]))
                        func_vec = np.array([f.factor_func.CalProduct(i, j) for j in range(self.num_label)])
                        f.marginal[i] = np.multiply(b_vec, func_vec)
                    f.marginal = f.marginal / np.sum(f.marginal)
                else:
                    for a in range(self.num_label):
                        for b in range(self.num_label):
                            b_vec = f.belief[0][a] * f.belief[1][b] * f.belief[2]
                            func_vec = np.array([f.factor_func.CalProduct(a, b, c) for c in range(self.num_label)])
                            f.marginalTD[a][b] = np.multiply(b_vec, func_vec)
                    f.marginalTD = f.marginalTD / np.sum(f.marginalTD)

    def maxSumPropagation(self, max_iter):
        """ This is the algorithm!
                            Each timestep:
                            take incoming messages and multiply together to produce outgoing for all nodes
                            then push outgoing to neighbors' incoming
                            check outgoing v. previous outgoing to check for convergence
                        """
        # loop to convergence
        timestep = 0
        while timestep < max_iter and not self.converged:  # run for maxsteps cycles
            timestep = timestep + 1

            for k, f in iteritems(self.fac):
                # start with factor-to-variable
                # can send immediately since not sending to any other factors
                self.diff_max = max([f.MaxSumPropagation(self.diff_max, self.labeled_given), self.diff_max])
                if self.diff_max < 1e-6:
                    break

            for k, v in iteritems(self.var):
                # variable-to-factor
                self.diff_max = max([v.MaxSumPropagation(self.diff_max, self.labeled_given), self.diff_max])
                if self.diff_max < 1e-6:
                    break

        #     # check for convergence
        #     t = True
        #     for k, v in iteritems(self.var):
        #         t = t and v.checkConvergence()
        #         if not t:
        #             break
        #     if t:
        #         for f in self.fac:
        #             t = t and f.checkConvergence()
        #             if not t:
        #                 break
        #
        #     if t:  # we have convergence!
        #         self.converged = True
        #
        # # if run for 500 steps and still no convergence:impor
        # if not self.converged:
        #     print("No convergence!")

    def bruteForce(self):
        """ Brute force method. Only here for completeness.
            Don't use unless you want your code to take forever to produce results.
            Note: index corresponding to var determined by order added
            Problem: max number of dims in numpy is 32???
            Limit to enabled vars as work-around
        """
        # Figure out what is enabled and save dimensionality
        enabledDims = []
        enabledNids = []
        enabledNames = []
        enabledObserved = []
        for k, v in iteritems(self.var):
            if v.enabled:
                enabledNids.append(v.nid)
                enabledNames.append(k)
                enabledObserved.append(v.observed)
                if v.observed < 0:
                    enabledDims.append(v.dim)
                else:
                    enabledDims.append(1)

        # initialize matrix over all joint configurations
        joint = np.zeros(enabledDims)

        # loop over all configurations
        self.configurationLoop(joint, enabledNids, enabledObserved, [])

        # normalize
        joint = joint / np.sum(joint)
        return {'joint': joint, 'names': enabledNames}

    def configurationLoop(self, joint, enabledNids, enabledObserved, currentState):
        """ Recursive loop over all configurations
            Used for brute force computation
            joint - matrix storing joint probabilities
            enabledNids - nids of enabled variables
            enabledObserved - observed variables (if observed!)
            currentState - list storing current configuration of vars up to this point
        """
        currVar = len(currentState)
        if currVar != len(enabledNids):
            # need to continue assembling current configuration
            if enabledObserved[currVar] < 0:
                for i in range(0, joint.shape[currVar]):
                    # add new variable value to state
                    currentState.append(i)
                    self.configurationLoop(joint, enabledNids, enabledObserved, currentState)
                    # remove it for next value
                    currentState.pop()
            else:
                # do the same thing but only once w/ observed value!
                currentState.append(enabledObserved[currVar])
                self.configurationLoop(joint, enabledNids, enabledObserved, currentState)
                currentState.pop()

        else:
            # compute value for current configuration
            potential = 1.
            for f in self.fac:
                if f.enabled and False not in [x.enabled for x in f.nbrs]:
                    # figure out which vars are part of factor
                    # then get current values of those vars in correct order
                    args = [currentState[enabledNids.index(x.nid)] for x in f.nbrs]

                    # get value and multiply in
                    potential = potential * f.P[tuple(args)]

            # now add it to joint after correcting state for observed nodes
            ind = [currentState[i] if enabledObserved[i] < 0 else 0 for i in range(0, currVar)]
            joint[tuple(ind)] = potential

    def marginalizeBrute(self, brute, var):
        """ Util for marginalizing over joint configuration arrays produced by bruteForce
        """
        sumout = list(range(0, len(brute['names'])))
        del sumout[brute['names'].index(var)]
        marg = np.sum(brute['joint'], tuple(sumout))
        return marg / np.sum(marg)  # normalize to sum to one
