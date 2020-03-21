# -*- coding: utf-8 -*-

from urllib import parse
import os
os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'My_SECRET_KEY'

    #是否启用mongodb
    MongoDbEnable = False
    MongoDbAuth = False
    MongoDbUsername = parse.quote_plus('sa')
    MongoDbPassword = parse.quote_plus('pass@word1')
    MongoDbHost = '127.0.0.1'
    MongoDbPort = '27017'

    #新闻摘要长度
    SummaryLength = 5

    #文件引用
    NewsFile = 'sqlResult_1558435.csv'
    #WordsModelFile = 'zhwiki_news.FastText.model'
    #WordsModelFile = 'zhwiki_news.word2vec.model'
    WordsModelFile = 'zhwiki_news.word2vec_min_count5.model'

    #模型处理方法
    ModelMethod = 'Word2Vec'
    #ModelMethod = 'FastText'