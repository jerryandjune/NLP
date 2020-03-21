# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:16:35 2020

@author: Jerry
"""

import pandas as pd

# 把处理好的中文wiki和新闻语料合拼到txt中
def concat_txt():
    # 读取中文wiki语料
    corpus_zhwiki = pd.read_csv('04_corpus.zhwiki.segwithb.txt',names='content')  
    # 读取新闻语料
    corpus_news = pd.read_csv('05_corpus.news.segwithb.txt',names='content')    
    # 用pd.concat把两个dataframe合拼一起
    corpus_zhwiki_news = pd.concat([corpus_zhwiki, corpus_news])    
    # 删除corpus_zhwiki, corpus_news以免占用内存
    del corpus_zhwiki, corpus_news
    # 输入合拼后的文件corpus.zhwiki_news.segwithb.txt，该文件可以直接用于训练词向量
    corpus_zhwiki_news.to_csv('06_corpus.zhwiki_news.segwithb.txt',header=None,index=None)


if __name__ == '__main__':
    concat_txt()











