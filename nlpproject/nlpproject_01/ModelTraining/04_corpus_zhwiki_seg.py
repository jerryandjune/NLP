# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:19:54 2020

@author: Jerry
"""

import os, sys
import logging
from optparse import OptionParser
import jieba
import os 
import sys

# 创建停用词列表，chinese_stopwords中有常用的中文停用词1800多个
def getStopwords():
    stopwords = [line.strip() for line in open('chinese_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

# 中文分词及去中文停用词主函数
def seg_with_jieba(infile, outfile,stopword=False):
    '''segment the input file with jieba'''
    with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', encoding='utf-8') as fout:
        i = 0       
        for line in fin:
            # jieba分词
            seg_list = jieba.cut(line)           
            sentence_segment=[] 
            # 遍历seg_list
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

# 程序主函数
def seg_with_jieba_main():
    # # 解析参数
    parser = OptionParser()
    # infile中default中添加wiki已处理繁体转简体，并去除多余字符的txt文件名称
    parser.add_option('-i','--input',dest='infile',default='03_corpus.zhwiki.simplified.done.txt',help='input file to be segmented')
    # outfile为输出处理好的语料文件名称，默认corpus.zhwiki.segwithb.txt
    parser.add_option('-o','--output',dest='outfile',default='04_corpus.zhwiki.segwithb.txt',help='output file segmented')
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])
    infile = options.infile
    outfile = options.outfile

    try:
        # 处理语料seg_with_jieba函数，接收的参数为parser中的infile, outfile设置的参数
        seg_with_jieba(infile,outfile)
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
    # 获取中文停用词列表
    stopwords = getStopwords()
    
    # 运行程序主函数
    seg_with_jieba_main()