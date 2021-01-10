

import datetime
import numpy as np
from tree_parent.cosineSim import cos_sim
from sklearn.decomposition import PCA

def strTransVector(array):
    x = array[1].split(',')
    currLineListFloat = []
    for i in x:
        # currLineListFloat.append(float(i))
        currLineListFloat.append(int(i))
    return currLineListFloat

def getMeta(array):
    compyAndGa = array[0]
    return compyAndGa



# 原始数据集计算相似度
if __name__ == "__main__":

    for readline in open("E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\resultVector.txt"):
        array = readline.strip('\n').split('$')
        compyAndGa1 = getMeta(array)
        vector1 = strTransVector(array)
        i = datetime.datetime.now()
        print(compyAndGa1, "当前的日期和时间是 %s" % i)
        for y in open("E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\resultVector.txt"):
            array = y.strip('\n').split('$')
            compyAndGa2 = getMeta(array)
            vector2 = strTransVector(array)
            content = compyAndGa1 + ',' + compyAndGa2 + ',' + cos_sim(vector1,vector2)
            print(content)

            # saveSimfile("E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\finalSimResult.txt", content)

