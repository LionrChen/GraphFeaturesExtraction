#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 16:47
# @Author  : ChenSir
# @File    : generateEdgesList.py
# @Software: PyCharm

import csv

# origin file
data_file = open("../dataset/digg_friends.csv", 'r')
csv_read_file = csv.reader(data_file)

# The file use to write extracted features.
output_file = open("../dataset/digg_friends_format.csv", 'w')
csv_write = csv.writer(output_file, dialect='excel')

for line in csv_read_file:
    if line[2] and line[3] is not None:
        csv_write.writerow(line)
