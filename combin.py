#-*- coding:utf-8 -*-

'''
@author: xllg
@project: AttReader
@file: combin.py
@time: 19-5-1 下午2:58
'''

#coding=utf-8
import os
#获取目标文件夹的路径
filedir = '/home/xllg/PycharmProjects/AttReader/data/datasets/dev'
#获取当前文件夹中的文件名称列表
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open('/home/xllg/PycharmProjects/AttReader/data/datasets/dev/dev.processed.dataset.txt','w')
#先遍历文件名
for filename in filenames:
    filepath = filedir+'/'+filename
    #遍历单个文件，读取行数
    for line in open(filepath):
        f.writelines(line)
    # f.write('\n')
#关闭文件
f.close()