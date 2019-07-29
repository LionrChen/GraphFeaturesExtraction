#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/26 18:27
# @Author  : ChenSir
# @File    : edge.py
# @Software: PyCharm


class Edge(object):
    def __init__(self, weight, head, tail):
        self.weight = weight
        self.head = head
        self.tail = tail
