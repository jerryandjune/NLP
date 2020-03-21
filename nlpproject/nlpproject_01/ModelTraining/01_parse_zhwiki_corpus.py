# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:34:35 2020

@author: Jerry
"""
# logging主要用于输出运行日志
import logging
import os.path
import sys
# optparse主要用于添加参数设置
from optparse import OptionParser
# gensim中专门处理WikiCorpus的类
from gensim.corpora import WikiCorpus


# 处理WikiCorpus函数，infile, outfile参数在需要在parser = OptionParser()中添加
def parse_corpus(infile, outfile):
    '''parse the corpus of the infile into the outfile'''
    space = ' '
    i = 0
    with open(outfile, 'w', encoding='utf-8') as fout:
        # gensim中的维基百科处理类WikiCorpus
        wiki = WikiCorpus(infile, lemmatize=False, dictionary={})  
        for text in wiki.get_texts():
            fout.write(space.join(text) + '\n')
            i += 1
            if i % 10000 == 0:
                logger.info('Saved ' + str(i) + ' articles')

# 程序主函数
def wiki_corpus_main():
        # 返回当前运行的py文件名称
    program = os.path.basename(sys.argv[0])
    # logging.basicConfig函数中，可以指定日志的输出格式format，这个参数可以输出很多有用的信息
    # %(asctime)s: 打印日志的时间     %(levelname)s: 打印日志级别名称      %(message)s: 打印日志信息
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
    # logging.getLogger(name)方法进行初始化，name可以不填。通常logger的名字我们对应模块名
    logger = logging.getLogger(program)  # logging.getLogger(logger_name)
    # logger.info打印程序运行是的正常的信息，用于替代print输出
    logger.info('running ' + program + ': parse the chinese corpus')

    # 解析参数
    parser = OptionParser()
    # infile中default中添加wiki未处理过的压缩包文件名称
    parser.add_option('-i','--input',dest='infile',default='zhwiki-20191120-pages-articles-multistream.xml.bz2',help='input: Wiki corpus')
    # outfile为输出处理好的语料文件名称，默认corpus.zhwiki.txt
    parser.add_option('-o','--output',dest='outfile',default='corpus.zhwiki.txt',help='output: Wiki corpus')
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])

    infile = options.infile
    outfile = options.outfile
#    infile = 'zhwiki-20191120-pages-articles-multistream.xml.bz2'
#    outfile = 'corpus.zhwiki.txt' 
    try:
        # 处理语料parse_corpus函数，接收的参数为parser中的infile, outfile设置的参数
        parse_corpus(infile, outfile)
        logger.info('Finished Saved ' + str(i) + 'articles')
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
    logger.info('running ' + program + ': parse the chinese corpus')
    
    # 运行程序主函数
    wiki_corpus_main()


    
        