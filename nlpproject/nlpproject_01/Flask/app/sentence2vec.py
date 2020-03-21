# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:27:20 2020

@author: Jerry
"""
from flask import Flask
import jieba
import re
import numpy as np
import pandas as pd
import gensim.models.base_any2vec
from scipy.spatial.distance import cosine
import os
from gensim.models import FastText
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import time
from functools import wraps
import gc
from app.models.Config import Config


class sentence2vec(object):
    # 解决matplotlib中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    REAL = np.float32

    # 定义预先计算归一化常数Z
    Z = 0.

    app = Flask(__name__)
    path = os.path.join(app.static_folder, Config.WordsModelFile)

    if Config.ModelMethod == 'Word2Vec':
        model = Word2Vec.load(path)
    elif Config.ModelMethod == 'FastText':
        model = FastText.load(path)
    else:
        model = Word2Vec.load(path)

    # 加载模型
    @staticmethod
    def init():
        # 预先计算归一化常数Z
        sentence2vec.Z = sentence2vec.normalization_constant_Z()
        pass

    @staticmethod
    def timefn(fn):
        """计算性能的修饰器"""
        @wraps(fn)
        def measure_time(*args, **kwargs):
            t1 = time.time()
            result = fn(*args, **kwargs)
            t2 = time.time()
            print("@timefn：" + fn.__name__ + "  生成摘要时间: " +
                np.str('%.2f' % (np.float32(t2 - t1))) + " 秒")
            return result
        return measure_time

    @staticmethod
    def normalization_constant_Z():
        '''计算归一化常数Z'''
        vlookup = sentence2vec.model.wv.vocab
        Z = 0
        for k in vlookup:
            Z += vlookup[k].count
        return Z

    @staticmethod
    def sif_embeddings(sentences, model, alpha=1e-3):
        """计算句子向量的SIF嵌入参数
        ----------
        sentences : list
            需要计算的句子或文章
        model : word2vec或FastText训练得到的模型
            一个包含词向量和词汇表的gensim模型
        alpha : float, optional
            参数，用于根据每个单词的概率p(w)对其进行加权。
        Returns
        -------
        numpy.ndarray 
            SIF 句子嵌入矩阵 len(sentences) * dimension
        """

        vlookup = model.wv.vocab  # 获取字典索引
        vectors = model.wv        # 我们能够访问词向量
        size = model.vector_size  # 词向量维度

        output = []

        # 遍历所有的句子
        for s in sentences:
            count = 0
            v = np.zeros(size, dtype=sentence2vec.REAL)  # 摘要向量
            # 遍历所有单词
            for w in s:
                # 单词必须出现在词汇表中
                if w in vlookup:
                    for i in range(size):
                        # 平滑逆频率，SIF
                        v[i] += (alpha / (alpha + (vlookup[w].count / sentence2vec.Z))
                                ) * vectors[w][i]
                    count += 1

            if count > 0:
                for i in range(size):
                    v[i] *= 1/count
            output.append(v)
        return np.vstack(output).astype(sentence2vec.REAL)

    @staticmethod
    def cut(text):
        '''分词函数'''
        return ' '.join(jieba.cut(text))
    
    @staticmethod
    def split_sentences(text):
        '''分句函数'''
        sents = []
        text = re.sub(r'\n+','。',text)   # 换行改成句号（标题段无句号的情况）
        text = re.sub('([。！？\?])([^’”])',r'\1\n\2',text)  # 普通断句符号且后面没有引号
        text = re.sub('(\.{6})([^’”])',r'\1\n\2',text)   # 英文省略号且后面没有引号
        text = re.sub('(\…{2})([^’”])',r'\1\n\2',text)   # 中文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',r'\1\n\2',text)  # 断句号+引号且后面没有引号    
        text = text.replace(u'。。',u'。')  # 删除多余的句号
        text = text.replace(u'？。',u'。')  #
        text = text.replace(u'！。',u'。')  # 删除多余的句号
        text = text.replace(u'\n', u'').replace(u'\r', u'')  # 删除多余的\\r\\n
        text = text.replace(u'\u3000',u'')
        text = text.replace(u'\\n',u'')
        text = text.replace(u'点击图片',u'')
        text = text.replace(u'进入下一页',u'')
        #sentences = re.split(r'。|！|？|】|；',text) # 分句
        sentences = re.split('。|！|\!|\.|？|\?',text) # 分句
        #sentences = re.split(r'[。，？！：]',text) # 分句
        sentences = sentences[:-1] # 删除最后一个句号后面的空句
        for sent in sentences:
            len_sent = len(sent)
            if len_sent < 4:  # 删除换行符、一个字符等
                continue
            # sent = sent.decode('utf8')
            sent = sent.strip('　 ')
            sent = sent.lstrip('【')
            sent = sent.lstrip('】')
            sents.append(sent)
        return sents

    @staticmethod
    def knn_smooth(arr):
        '''knn平滑函数'''
        result = []
        if len(arr) >3:
            result = []
            for i in range(len(arr)):
                a = 0
                # 处理第一句余弦距离时，取第一，第二句的余弦距离之和，再取平均，作为第一句的余弦距离
                if i < 1: 
                    a = ((arr[i] + arr[i+1])/2)
                    result.append(a)
                # 处理中间句子余弦距离时，取前一句，当前句，后一句的余弦距离之和，再取平均，作为的余弦距离
                elif i<len(arr)-1:
                    a = ((arr[i] + arr[i - 1]+ arr[i + 1]) / 3) 
                    result.append(a)
                # 处理最后一句余弦距离时，取最后一句，前一句的余弦距离之和，再取平均，作为最后一句的余弦距离
                else:
                    a = ((arr[i] + arr[i - 1]) / 2) 
                    result.append(a)
        else:
            result = arr
        return result
    
    @staticmethod
    def get_plot(x1,x2,top_n):
        plt.figure(figsize=(12,8))
        plt.plot(x1[:top_n],linestyle = '-.',marker = 'o',color='r',alpha = 0.5,label='平滑前')
        plt.plot(x2[:top_n],linestyle = '-.',marker = 'o',color='g',alpha = 0.5,label='平滑后')
        plt.title('K N N连续句子相关性的平滑')
        plt.xlabel('句子编号')
        plt.ylabel('余弦距离(数值越小，句子越重要)')
        plt.grid(linestyle='-.',alpha=0.7)
        plt.legend()
        for i,j in zip(np.arange(len(x1[:top_n])),x1[:top_n]):
            plt.text(i,j+0.002,'%.3f' % j, color = 'r',alpha = 0.7)
        for i,j in zip(np.arange(len(x2[:top_n])),x2[:top_n]):
            plt.text(i,j+0.002,'%.3f' % j, color = 'g',alpha = 0.7)     
    
    @staticmethod
    def get_sen_doc_cosine(text,title,top_n=10,plot=True):
        '''获取 句向量/文章向量 的余弦距离'''
        # 判断对象是否list
        if isinstance(text,list): text = ' '.join(text)
        # 文章分句
        split_sens = sentence2vec.split_sentences(text)
        # 文章向量化,标题向量化
        doc_vec = sentence2vec.sif_embeddings([text], sentence2vec.model, alpha=1e-3)
        # 定义句子/文章向量  和  句子/标题向量 余弦距离空字典
        sen_doc_cosine = {} 
        # 遍历文章分句，计算句向量，把文章的内容和对应的余弦距离存入字典
        for sen in split_sens:
            sen_vec = sentence2vec.sif_embeddings([sen], sentence2vec.model, alpha=1e-3)
            # 计算 句子/文章向量 的余弦距离
            sen_doc_cosine[sen] = cosine(sen_vec, doc_vec)           
        # 句子/文章向量 余弦字典的keys，values空列表
        sen_doc_cosine_keys ,sen_doc_cosine_values = [] , [] 
        # 遍历句子/文章向量 余弦距离字典，获取正确的分句内容和对应的余弦距离存入对应列表中    
        for i,j in sen_doc_cosine.items():
            sen_doc_cosine_keys.append(i)
            sen_doc_cosine_values.append(j)
        # 平滑前, 把（句子/文章向量）列表转成数组
        knn_before_cosine_values = np.array(sen_doc_cosine_values)
        # 使用自定义的knn_smooth函数，计算新的余弦距离 （平滑后的余弦距离） 
        knn_after_cosine_values = np.array(sentence2vec.knn_smooth(sen_doc_cosine_values))
        # 定义knn平滑后的余弦距离空字典
        knn_cosine_score = {}
        # 把原分句内容和平滑后的余弦距离组合成字典
        knn_cosine_score = dict(zip(sen_doc_cosine_keys,knn_after_cosine_values))
        # 绘制平滑前后的余弦距离的曲线图
        if plot:
            sentence2vec.get_plot(knn_before_cosine_values, knn_after_cosine_values, top_n)  
        # 返回经过平滑后的字典，降序,字典包含新闻分句和对应的余弦距离
        return sorted(knn_cosine_score.items(), key=lambda x:x[1], reverse=False)
    
    # 有输入标题
    @staticmethod
    def get_sen_doc_title_cosine(text,title,weight = 0.5,top_n=10,plot=True):
        '''获取（句子/文章向量）（句子/标题向量）的余弦距离'''
        # 判断对象是否list
        if isinstance(text,list): text = ' '.join(text)
        # 文章分句
        split_sens = sentence2vec.split_sentences(text)
        # 文章向量化,标题向量化
        doc_vec = sentence2vec.sif_embeddings([text], sentence2vec.model, alpha=1e-3)
        title_vec = sentence2vec.sif_embeddings([title], sentence2vec.model, alpha=1e-3)
        # 定义句子/文章向量  和  句子/标题向量 余弦距离空字典
        sen_doc_cosine, sen_title_cosine = {} , {}
        # 遍历文章分句，计算句向量，把文章的内容和对应的余弦距离存入字典
        for sen in split_sens:
            sen_vec = sentence2vec.sif_embeddings([sen], sentence2vec.model, alpha=1e-3)
            # 计算 句子/文章向量 的余弦距离
            sen_doc_cosine[sen] = cosine(sen_vec, doc_vec)
            # 计算 句子/标题向量 的余弦距离
            sen_title_cosine[sen] = cosine(sen_vec, title_vec)             
        # 句子/文章向量 余弦字典的keys，values空列表
        sen_doc_cosine_keys ,sen_doc_cosine_values = [] , [] 
        # 遍历句子/文章向量 余弦距离字典，获取正确的分句内容和对应的余弦距离存入对应列表中    
        for i,j in sen_doc_cosine.items():
            sen_doc_cosine_keys.append(i)
            sen_doc_cosine_values.append(j)
        # 句子/标题向量 余弦字典的keys，values空列表
        sen_title_cosine_keys ,sen_title_cosine_values= [] , []
        # 遍历 句子/标题向量 余弦距离字典，获取正确的分句内容和对应的余弦距离存入对应列表中    
        for i,j in sen_title_cosine.items():
            sen_title_cosine_keys.append(i)
            sen_title_cosine_values.append(j)
        # 平滑前,计算 （句子/文章向量）* 权重  + （句子/标题向量）* （1 - 权重)
        knn_before_cosine_values = np.array(sen_doc_cosine_values) * weight + np.array(sen_title_cosine_values) * (1 - weight)       
        # 使用自定义的knn_smooth函数，计算新的余弦距离 （平滑后的余弦距离） 
        knn_after_cosine_values = np.array(sentence2vec.knn_smooth(sen_doc_cosine_values)) * weight + np.array(sentence2vec.knn_smooth(sen_title_cosine_values)) * (1 - weight)
        # 定义knn平滑后的余弦距离空字典
        knn_cosine_score = {}
        # 把原分句内容和平滑后的余弦距离组合成字典
        knn_cosine_score = dict(zip(sen_doc_cosine_keys,knn_after_cosine_values))
        # 绘制平滑前后的余弦距离的曲线图
        if plot:
            sentence2vec.get_plot(knn_before_cosine_values, knn_after_cosine_values, top_n)  
        # 返回经过平滑后的字典，降序,字典包含新闻分句和对应的余弦距离
        return sorted(knn_cosine_score.items(), key=lambda x:x[1], reverse=False)
    
    @staticmethod
    def get_summarize(text, title, weight = 0.5, top_n = 10 ,plot = False):             
        '''生成摘要，默认获得前10句'''
        # 获取分句
        split_sens = sentence2vec.split_sentences(text)
        # 获取排序后的字典，key为句子内容，values为分句向量与文章向量的余弦距离
        if title =='':
            ranking_sentences = sentence2vec.get_sen_doc_cosine(text, title, top_n = top_n, plot = plot)
        else:
            ranking_sentences = sentence2vec.get_sen_doc_title_cosine(text, title, weight = weight, top_n = top_n, plot = plot)
        # 设置一个空集合和空字符
        selected_sen = set()
        if len(split_sens) > top_n:
            # 遍历top_n的句子，并添加到空集合       
            for sen, _ in ranking_sentences[:top_n]:
                selected_sen.add(sen)
        else:
            for sen, _ in ranking_sentences:
                selected_sen.add(sen)                
        # 设置摘要的空列表
        summarize = []
        # 遍历所有的句子，把top_n的句子，按照原新闻中的顺序拼接起来
        for sen in split_sens:
            if sen in selected_sen:
                summarize.append(sen+'。')
        summarize = ' '.join(summarize)
        return summarize