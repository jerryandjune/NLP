# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:55:55 2020

@author: Jerry
"""
import jieba
import numpy as np
import pandas as pd
import time
from functools import wraps
import gc
import jieba.analyse
from collections import Counter
from flask import Flask,jsonify
def timefn(fn):
    """计算性能的修饰器"""
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn：" + fn.__name__ + "  生成关键字时间: " + np.str('%.2f'%(np.float32(t2 - t1))) + " 秒")
        return result
    return measure_time


#--------------基于word2vec提取关键字，由于太慢，不考虑使用------------
#此函数计算某词对于模型中各个词的转移概率p(wk|wi)
def predict_proba(oword, iword):
    #获取输入词的词向量
    iword_vec = model.wv[iword]
    #获取保存权重的词的词库
    oword = model.wv.vocab[oword]
    oword_l = model.trainables.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot) 
    return lprob

def word2vec_keywords(s):
    #抽出s中和与训练的model重叠的词
    s = [w for w in s if w in model.wv]
    ws = {w:sum([predict_proba(u, w) for u in s]) for w in s}
    return Counter(ws).most_common()
#--------------基于word2vec提取关键字，由于太慢，不考虑使用------------
    

@timefn
def get_keywords(s,topK = 10):
    ''' 基于textrank算法提取关键字'''
    #词性限制集合为["ns", "n", "vn", "v", "nr"]，表示只能从词性为地名、名词、动名词、动词、人名这些词性的词中抽取关键词。
    allow_pos = ("ns", "n", "vn", "v", "nr")
    keywords = jieba.analyse.textrank(s, topK=topK, withWeight=True, allowPOS = allow_pos )
    keywords_item = []
    for kw,_ in keywords:
    # 只获取关键字
        keywords_item.append(kw)
    return jsonify({"keywords": keywords_item})


if __name__ == '__main__':
    # 使用pandas读取新闻语料作为测试数据
    news_data = pd.read_csv('sqlResult_1558435.csv')
    # 把新闻语料转成list
    news_content_list = news_data['content'].fillna('').to_list()
    # 删除新闻语料占用的内存空间
    del news_data
    # 释放内存
    gc.collect()
    # 随机在news_content_list中取一条新闻
    sens = np.random.choice(news_content_list[9:1000])    
    # 关键字提取
    keywords_result = get_keywords(sens,topK = 20)











