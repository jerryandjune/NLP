# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:26:05 2020

@author: Jerry
"""


import os, sys
import logging
import multiprocessing
from optparse import OptionParser
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence

# FastText训练词向量模型函数
def FastText_train(infile,outmodel,outvector,size,window,min_count,sg):
    '''train the word vectors by FastText'''
    
    # 训练模型
    model = FastText(LineSentence(infile),size=size,window=window,min_count=min_count,sg=sg,workers=multiprocessing.cpu_count())  

    # 保存模型
    model.save(outmodel)
    model.wv.save_word2vec_format(outvector,binary=False)


# word2vec程序主函数
def FastText_train_main():
    # 解析参数
    parser = OptionParser()
    # infile中default中添加需要训练词向量的txt文件名称
    parser.add_option('-i','--input',dest='infile',default='06_corpus.zhwiki_news.segwithb.txt',help='zhwiki corpus')
    # 设置保存模型的参数
    parser.add_option('-m','--outmodel',dest='wv_model',default='zhwiki_news.FastText.model',help='FastText model')
    parser.add_option('-v','--outvec',dest='wv_vectors',default='zhwiki_news.FastText.vectors',help='FastText vectors')
    # 设置训练模型的参数size，是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好
    parser.add_option('-s',type='int',dest='size',default=200,help='word vector size')
    # 设置训练模型的参数window，表示当前词与预测词在一个句子中的最大距离是多少,默认值为5
    parser.add_option('-w',type='int',dest='window',default=5,help='window size')
    # 设置训练模型的参数min_count，可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    parser.add_option('-n',type='int',dest='min_count',default=2,help='min word frequency')
    # 设置训练模型的参数sg，用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    parser.add_option('-g',type='int',dest='sg',default=1,help='sg')
    
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])

    infile = options.infile
    outmodel = options.wv_model
    outvec = options.wv_vectors
    vec_size = options.size
    window = options.window
    min_count = options.min_count
    sg = options.sg

    try:       
        FastText_train(infile, outmodel, outvec, vec_size, window, min_count,sg)
        logger.info('FastText model training finished')
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
    logger.info('running ' + program)    
    # 运行程序主函数
    FastText_train_main()