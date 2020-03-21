# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:06:04 2020

@author: Jerry
"""

import os, sys
import logging
from optparse import OptionParser
import re

# 处理去掉语料中英文、空格和多余字符主函数，infile, outfile参数在需要在parser = OptionParser()中添加
def remove_en_blank(infile,outfile):
    '''remove the english word and blank from infile, and write into outfile'''
    with open(infile,'r',encoding='utf-8') as fin, open(outfile,'w',encoding='utf-8') as fout:
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
            fout.write(res)
            i += 1
            if i % 1000 == 0:
                logger.info('handing with the {} line'.format(i))

# 程序主函数
def remove_en_blank_main():
    # 解析参数
    parser = OptionParser()
    # infile中default中添加wiki处理过的繁体转简体的txt文件名称
    parser.add_option('-i','--input',dest='infile',default='02_corpus.zhwiki.simplified.txt',help='input file to be preprocessed')
    # outfile为输出处理好的语料文件名称，默认corpus.zhwiki.simplified.done.txt
    parser.add_option('-o','--output',dest='outfile',default='03_corpus.zhwiki.simplified.done.txt',help='output file removed english and blank')
    # 初始化args为空list，否则在jupyter notebook中会报错
    args = []
    # 读取parser中的参数
    (options,args) = parser.parse_args(args = [])

    infile = options.infile
    outfile = options.outfile

    try:
        # 处理语料remove_en_blank函数，接收的参数为parser中的infile, outfile设置的参数
        remove_en_blank(infile, outfile)
        logger.info('remove english and blank finished')
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
    logger.info('running ' + program + ': remove english and blank')    
    # 运行程序主函数
    remove_en_blank_main()
    
    
    
    