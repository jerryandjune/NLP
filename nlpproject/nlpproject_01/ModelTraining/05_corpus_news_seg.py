# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:44:00 2020

@author: Jerry
"""

import os, sys
import logging
from optparse import OptionParser
import jieba
import os 
import re
import sys
import pandas as pd

# 读取新闻语料函数
def read_news_data():
    # 用pd.read_csv方法读取新闻语料
    df = pd.read_csv('sqlResult_1558435.csv',encoding = 'utf-8')
    # 把新闻语料中的content列，单独保存为corpus.news.txt
    df['content'].to_csv('corpus.news.txt',header=None,index=None)
    
    
# 创建停用词列表，chinese_stopwords中有常用的中文停用词1800多个
def getStopwords():
    stopwords = [line.strip() for line in open('chinese_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

# 新闻语料，中文分词及去中文停用词主函数
def seg_with_jieba(infile, outfile,stopword=False):
    '''segment the input file with jieba'''
    with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', encoding='utf-8') as fout:
        relu1 = re.compile(r'[ a-zA-Z]')  # delete english char and blank
        relu2 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        relu3 = re.compile(r'[（\(][，；。？！\s]*[）\)]')
        relu4 = re.compile(r'[「『]')
        relu5 = re.compile(r'[」』]')                
        i = 0      
        
        for line in fin:
            res = relu1.sub('', line)
            res = relu2.sub('\2', line)
            res = relu3.sub('', line)
            res = relu4.sub('“', line)
            res = relu5.sub('”', line) 
            # jieba分词
            seg_list = jieba.cut(line)            
            sentence_segment=[] 
            # 判断行里不为"
            if seg_list !='"':
                for word in seg_list:
                    if stopword:                        
                        # 判断word不在停用词列表中才执行append
                        if word not in stopwords:
                            sentence_segment.append(word)
                    sentence_segment.append(word)        
                # 把已去掉停用词的sentence_segment，用' '.join()拼接起来
                seg_res = ' '.join(sentence_segment)
                # 把拼接好的分词结果，写入output文件
                fout.write(seg_res)
                i += 1
                if i % 1000 ==0:
                    logger.info('handing with {} line'.format(i))


# 由于去除多余字符后，分词文件有许多空行，现在要去掉除语料中的空行
def remove_blank_lines(outfile,new_outfile):
    with open('corpus.news.segwithb_temp.txt','r',encoding = 'utf-8') as fr,open('05_corpus.news.segwithb.txt','w',encoding = 'utf-8') as fd:
        for text in fr.readlines():            
            if text.strip().split():
                    fd.write(text)
    print('输出成功....')


    
# 新闻语料主函数
def news_seg_main():
    # 解析参数
    parser = OptionParser()
    # infile中default中添加新闻语料txt文件名称，该文件在read_data()函数中产生
    parser.add_option('-i','--input',dest='infile',default='corpus.news.txt',help='input file to be segmented')
    # 临时存储已去掉多余字符的分词txt文件
    parser.add_option('-o','--output',dest='outfile',default='corpus.news.segwithb_temp.txt',help='output file segmented')
    # 输出处理好的分词文件，默认corpus.news.segwithb.txt
    parser.add_option('-n','--new_output',dest='new_outfile',default='05_corpus.news.segwithb.txt',help='new_output file segmented')
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])
    
    infile = options.infile
    outfile = options.outfile
    new_outfile = options.new_outfile
    # 读取新闻语料
    read_news_data()
    # 获取中文停用词列表
    stopwords = getStopwords()
    try:
        # 处理语料seg_with_jieba函数，接收的参数为parser中的infile, outfile设置的参数
        seg_with_jieba(infile,outfile)
        # 运行去空行函数
        remove_blank_lines(outfile,new_outfile)
        logger.info('segment the infile finished')
    except Exception as err:
        logger.info(err)    
    
if __name__ == '__main__':
    # 返回当前运行的py文件名称
    program = os.path.basename(sys.argv[0])
    # logging.basicConfig函数中，可以指定日志的输出格式format，这个参数可以输出很多有用的信息
    # %(asctime)s: 打印日志的时间     %(levelname)s: 打印日志级别名称      %(message)s: 打印日志信息
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
    # logging.getLogger(name)方法进行初始化，name可以不填。通常logger的名字我们对应模块名
    logger = logging.getLogger(program)  # logging.getLogger(logger_name)
    # logger.info打印程序运行是的正常的信息，用于替代print输出
    logger.info('running ' + program + ': segmentation of corpus by jieba')
    news_seg_main()
    
    
    
    
    