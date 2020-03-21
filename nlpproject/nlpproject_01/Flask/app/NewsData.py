# -*- coding: utf-8 -*-

from flask import Flask
import pandas as pd
import gc
import random
from app.models.Forms import NewsForm
import os
from app.models.Config import Config


class NewsData(object):
    news_content_list = []
    news_title_list = []

    @staticmethod
    def init():
        app = Flask(__name__)
        path = os.path.join(app.static_folder, Config.NewsFile)
        news_data = pd.read_csv(path)    #, encoding='gb18030'
        # 筛选新闻内容字符串长度大于500的新闻
        news_data = news_data[news_data['content'].str.len() > 500]
        # 把新闻语料转成list
        NewsData.news_content_list = news_data['content'].fillna('').to_list()
        NewsData.news_title_list = news_data['title'].fillna('').to_list()
        # 删除新闻语料占用的内存空间
        del news_data
        # 释放内存
        gc.collect()

    @staticmethod
    def GetNewData():

        #随机获取ID
        id = random.randint(0, len(NewsData.news_content_list) - 1)
        org = NewsData.news_content_list[id]
        # 处理文章中的\n字符
        NewsContent = str(org).replace('\\n', '')
        # 处理文章中的\u3000字符
        NewsContent = str(NewsContent).replace('\u3000', '')
        return NewsForm(
            NewsTitle=NewsData.news_title_list[id],
            NewsContent=NewsContent
        )
