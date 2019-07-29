#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/26 16:09
# @Author  : ChenSir
# @File    : SCC.py
# @Software: PyCharm


# reverse graph
def tr(G):
    GT = dict()
    for u in G.keys():
        GT[u] = GT.get(u, set())
    for u in G.keys():
        for v in G[u]:
            GT[v].add(u)
    return GT


# gain a sequence sorted by time
def topoSort(G):
    res = []
    S = set()

    # Depth-first traversal/search
    def dfs(G, u):
        if u in S:
            return
        S.add(u)
        for v in G[u]:
            if v in S:
                continue
            dfs(G, v)
        res.append(u)

    # check each node was leave out
    for u in G.keys():
        dfs(G, u)
    res.reverse()
    return res


# gain singe strongly connected components with assigns start node
def walk(G, s, S=None):
    if S is None:
        s = set()
    Q = []
    P = dict()
    Q.append(s)
    P[s] = None
    while Q:
        u = Q.pop()
        for v in G[u]:
            if v in P.keys() or v in S:
                continue
            Q.append(v)
            P[v] = P.get(v, u)
    return P


# a sample of graph, use dict to store graph.
# G = {
#     'a': {'b', 'c'},
#     'b': {'d', 'e', 'i'},
#     'c': {'d'},
#     'd': {'a', 'h'},
#     'e': {'f'},
#     'f': {'g'},
#     'g': {'e', 'h'},
#     'h': {'i'},
#     'i': {'h'}
# }


def SCC(G):
    seen = set()
    # to store scc
    scc = []
    GT = tr(G)
    for u in topoSort(G):
        if u in seen:
            continue
        C = walk(GT, u, seen)
        seen.update(C)
        scc.append(sorted(list(C.keys())))
    return scc
