#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 17:35
# @Author  : ChenSir
# @File    : test_deepwalk.py
# @Software: PyCharm

from deep_walk.deepwalk import load_edgelist, sample_walks
from gensim.models import Word2Vec

walk_length = 80
number_walks = 15
g = load_edgelist("../dataset/facebook_combined.txt")
total_walks = sample_walks(g, walk_length=walk_length, number_walks=number_walks)


emb_size = 128
window_size = 10
model_w2v = Word2Vec(sentences=total_walks, sg=1, hs=0, size=emb_size, window=window_size,\
                     min_count=0, workers=10, iter=10)
print(model_w2v.wv)