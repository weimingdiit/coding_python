import numpy as np

from Tools import saveSimfile
from tree_parent.Pca import fun3, fun0, fun4
from tree_parent.SimTree import getMeta, strTransVector
from tree_parent.cosineSim import cosine_similarity, cos_sim


def initZeroMatrix(size):
    sourceZero = []
    for i in range(1, size + 1):
        sourceZero.append(0)
    return sourceZero


'''
value:原始数据，
m:需要降维的目标维度
n：需要拼接0数组的个数
m<n && m<len(value)
'''


def readyData(value, m, n):
    # test data
    # mat = [[1,1,0,1,1],[1,1,0,1,1],[1,0,1,1,0]]
    # mat = [[1,1,0,1,1],[1,0,0,1,1],[1,0,1,1,0]]
    # mat = [[1,1,0,1,1,1,0,0,1,1,1,0,1,1,0]]

    mat = []
    sourceZero = initZeroMatrix(len(value))
    mat.append(value)
    # 初始化全0 矩阵
    for i in range(0, n):
        mat.append(sourceZero)

    # simple transform of test data
    Mat = np.array(mat, dtype='float64')
    print('Before PCA transforMation, data is:\n', Mat)
    print('\nMethod 1: PCA by original algorithm:')
    p, n = np.shape(Mat)  # shape of Mat
    t = np.mean(Mat, 0)  # mean of each column

    # substract the mean of each column
    for i in range(p):
        for j in range(n):
            Mat[i, j] = float(Mat[i, j] - t[j])
    # covariance Matrix //协方差
    cov_Mat = np.dot(Mat.T, Mat) / (p - 1)
    # fun1(Mat, cov_Mat, n)
    # fun2(Mat, cov_Mat)

    return fun3(mat, m)


'''
将  resultVector.txt 文件中的高维数据进行降维
结果写入 dimensionalityReduction.txt
'''


def dimensionalityReductionTest(input, output):
    for readline in open(input):
        array = readline.strip('\n').split('$')
        compyAndGa1 = getMeta(array)
        vector1 = strTransVector(array)
        # 拼接0 元素之后采用pca降维
        result = readyData(vector1, 4, 5)
        content = compyAndGa1 + ',' + str(result)
        saveSimfile(output, content)


def dimensionalityReductionReal(input, output):
    for readline in open(input):
        array = readline.strip('\n').split('$')
        compyAndGa1 = getMeta(array)
        vector1 = strTransVector(array)
        # 不需要拼接0 元素，直接采用pca降维
        result = fun4(vector1, 10)
        content = compyAndGa1 + '@' + str(result.tolist())
        print(content)
        saveSimfile(output, content)


# def juzhenTransList(result):


if __name__ == "__main__":
    inputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\resultVector.txt"
    outputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\dimensionalityReduction.txt"
    # dimensionalityReductionTest(inputPath,outputPath)
    dimensionalityReductionReal(inputPath, outputPath)

    # 生成字典，加载到内存，提高计算效率
    dict = {}
    for readline in open(outputPath):
        array = readline.strip('\n').split('@')
        dlist = array[1].replace('[', '').replace(']', '').split(',')
        currLineListFloat = []
        for i in dlist:
            if (i != ''):
                currLineListFloat.append(float(i))
        dict[array[0]] = currLineListFloat

    outputFinal = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\finalSimResult.txt"
    for key1 in dict:
        for key2 in dict:
            value = cos_sim(dict[key1], dict[key2])
            print(key1, key2, value)
            content = key1 + ',' + key2 + ',' + value
            saveSimfile(outputFinal, content)
            # valuefun2 = cosine_similarity(dict[key1], dict[key2], False)
            # print(key1, key2, valuefun2)
