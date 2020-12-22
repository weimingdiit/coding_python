#!usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@author: ming.wei
"""
import csv
import pickle


# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


def saveSimfile(savepath, content):
    with open(savepath, mode='a') as fp:
        fp.write(content)
        fp.write('\n')

# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 读取csv文件
def readCsv(path):
    with open(path, 'r') as f:
        csv_rows = list(csv.reader(f))
    return csv_rows


# csv文件每一行保存为一个文件
def transCsvTotxt(path, txt_path):
    csv_rows = readCsv(path)
    for row in csv_rows:
        savefile(txt_path + row[0], bytes(row[1], "UTF-8"))


# csv文件转为 text[]
def transCsvTotxtArray(path):
    texts = []
    csv_rows = readCsv(path)
    for row in csv_rows:
        texts.append(row[2])
    return texts


# csv文件转为 text[]
def transCsvTotxtMap(path):
    dict = {}
    csv_rows = readCsv(path)
    for row in csv_rows:
        dict[row[0]] = row[1]
    return dict


def transCsvToMap(path):
    dict = {}
    csv_rows = readCsv(path)
    for row in csv_rows:
        dict[row[0]] = row[1]
    return dict

# csv文件转为企业名单
def transCsvToCompanyName(path):
    dict = {}
    csv_rows = readCsv(path)
    for index, row in enumerate(csv_rows):
        dict[index] = row[0] + "_" + row[1]
    return dict
