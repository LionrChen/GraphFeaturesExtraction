#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 16:44
# @Author  : ChenSir
# @File    : bulid_graphAsNumpy.py
# @Software: PyCharm

import time
import numpy as np
import pandas as pd
import igraph as ig
start1 = time.time()
data_file_np = np.loadtxt("../dataset/digg_friends_format.csv", delimiter=",", usecols=(0, 2, 3)).astype(np.float32)
end1 = time.time()
start2 = time.time()
data_file_pd = pd.read_csv("../dataset/digg_friends_format.csv", usecols=(0, 2, 3), header=None)
end2 = time.time()
print("Time of numpy is {}, and time of pandas is {}".format(end1-start1,end2-start2))