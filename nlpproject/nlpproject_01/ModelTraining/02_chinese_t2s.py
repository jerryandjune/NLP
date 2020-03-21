# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:17:37 2020

@author: Jerry
"""

import os, sys
import logging
# optparse主要用于添加参数设置
from optparse import OptionParser
# opencc主要处理中文繁体转简体，安装命令pip install opencc-python-reimplemented
from opencc import OpenCC


# 处理繁体转简体主函数，infile, outfile参数在需要在parser = OptionParser()中添加
def zh_wiki_t2s(infile, outfile):
    '''convert the traditional Chinese of infile into the simplified Chinese of outfile'''
    # 读取已提取的wiki语料txt文件
    t_corpus = []
    with open(infile,'r',encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').replace('\t','')
            t_corpus.append(line)
    logger.info('read traditional file finished!')

    # convert the t_Chinese to s_Chinese
    cc = OpenCC('t2s')
    s_corpus = []
    for i,line in zip(range(len(t_corpus)),t_corpus):
        if i % 1000 == 0:
            logger.info('convert t2s with the {}/{} line'.format(i,len(t_corpus)))
        # s_corpus.append(OpenCC.convert(line))
        s_corpus.append(cc.convert(line))
    logger.info('convert t2s finished!')

    # 把已转换为简体的s_corpus，写入输出文件
    with open(outfile, 'w', encoding='utf-8') as f:
        for line in s_corpus:
            f.writelines(line + '\n')
    logger.info('write the simplified file finished!')

# 程序主函数
def t2s_main():   
    # 解析参数
    parser = OptionParser()
    # infile中default中添加wiki在步骤一中已经提取的txt文件
    parser.add_option('-i','--input',dest='input_file',default='01_corpus.zhwiki.txt',help='traditional file')
    # outfile为输出处理好的繁体转简体文件名称，默认corpus.zhwiki.simplified.txt
    parser.add_option('-o','--output',dest='output_file',default='02_corpus.zhwiki.simplified.txt',help='simplified file')
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])
    
    input_file = options.input_file
    output_file = options.output_file

    try:
        # 处理语料zh_wiki_t2s函数，接收的参数为parser中的infile, outfile设置的参数
        zh_wiki_t2s(infile=input_file,outfile=output_file)
        logger.info('Traditional Chinese to Simplified Chinese Finished')
    except Exception as err:
        logger.info(err)
        
        
if __name__ == '__main__':
    # 返回当前运行的py文件名称
    program = os.path.basename(sys.argv[0])
    # logging.getLogger(name)方法进行初始化，name可以不填。通常logger的名字我们对应模块名
    logger = logging.getLogger(program)
    # logging.basicConfig函数中，可以指定日志的输出格式format，这个参数可以输出很多有用的信息
    # %(asctime)s: 打印日志的时间     %(levelname)s: 打印日志级别名称      %(message)s: 打印日志信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger.info打印程序运行是的正常的信息，用于替代print输出
    logger.info('running ' + program + ' : convert Traditional Chinese to Simplified Chinese')
    # 运行程序主函数
    t2s_main()