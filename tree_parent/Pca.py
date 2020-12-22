import datetime

import numpy as np
from sklearn.decomposition import PCA
import sys


#returns choosing how many main factors
from Tools import saveSimfile
from tree_parent.SimTree import getMeta, strTransVector


def index_lst(lst, component=0, rate=0):
    #component: numbers of main factors
    #rate: rate of sum(main factors)/sum(all factors)
    #rate range suggest: (0.8,1)
    #if you choose rate parameter, return index = 0 or less than len(lst)
    if component and rate:
        print('Component and rate must choose only one!')
        sys.exit(0)
    if not component and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif component:
        print('Choosing by component, components are %s......'%component)
        return component
    else:
        print('Choosing by rate, rate is %s ......'%rate)
        for i in range(1, len(lst)):
            if sum(lst[:i])/sum(lst) >= rate:
                return i
        return 0

def fun1(Mat,cov_Mat,n):
    # PCA by original algorithm
    # eigvalues and eigenvectors of covariance Matrix with eigvalues descending
    U,V = np.linalg.eigh(cov_Mat)
    # Rearrange the eigenvectors and eigenvalues
    U = U[::-1]
    for i in range(n):
        V[i,:] = V[i,:][::-1]
    # choose eigenvalue by component or rate, not both of them euqal to 0
    Index = index_lst(U, component=2)  # choose how many main factors
    if Index:
        v = V[:,:Index]  # subset of Unitary matrix
    else:  # improper rate choice may return Index=0
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i])/sum(U) for i in range(1, len(U)+1)])
        sys.exit(0)
    # data transformation
    T1 = np.dot(Mat, v)
    # print the transformed data
    print('We choose %d main factors.'%Index)
    print('After PCA transformation, data becomes:\n',T1)


def fun2(Mat,cov_Mat):
    # PCA by original algorithm using SVD
    print('\nMethod 2: PCA by original algorithm using SVD:')
    # u: Unitary matrix,  eigenvectors in columns
    # d: list of the singular values, sorted in descending order
    u,d,v = np.linalg.svd(cov_Mat)
    Index = index_lst(d, rate=0.95)  # choose how many main factors
    T2 = np.dot(Mat, u[:,:Index])  # transformed data
    print('We choose %d main factors.'%Index)
    print('After PCA transformation, data becomes:\n',T2)

def fun3(mat,n):
    # PCA by Scikit-learn
    pca = PCA(n_components=n) # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    # print('\nMethod 3: PCA by Scikit-learn:')
    # print('After PCA transformation, data becomes:')
    result = pca.fit_transform(mat)  # transformed data
    # print(result)
    return result


def initZeroMatrix(size):
    sourceZero = []
    for i in range(1,size+1):
        sourceZero.append(0)
    return sourceZero

'''
value:原始数据，
m:需要降维的目标维度
n：需要拼接0数组的个数
m<n && m<len(value)
'''
def readyData(value,m,n):
    # test data
    # mat = [[1,1,0,1,1],[1,1,0,1,1],[1,0,1,1,0]]
    # mat = [[1,1,0,1,1],[1,0,0,1,1],[1,0,1,1,0]]
    # mat = [[1,1,0,1,1,1,0,0,1,1,1,0,1,1,0]]

    mat = []
    sourceZero = initZeroMatrix(len(value))
    mat.append(value)
    # 初始化全0 矩阵
    for i in range(0,n):
        mat.append(sourceZero)


    # simple transform of test data
    # Mat = np.array(mat, dtype='float64')
    # print('Before PCA transforMation, data is:\n', Mat)
    # print('\nMethod 1: PCA by original algorithm:')
    # p,n = np.shape(Mat) # shape of Mat
    # t = np.mean(Mat, 0) # mean of each column

    # substract the mean of each column
    # for i in range(p):
    #     for j in range(n):
    #         Mat[i,j] = float(Mat[i,j]-t[j])
    # covariance Matrix //协方差
    # cov_Mat = np.dot(Mat.T, Mat)/(p-1)
    # fun1(Mat, cov_Mat, n)
    # fun2(Mat, cov_Mat)

    return fun3(mat,m)

'''
将  resultVector.txt 文件中的高维数据进行降维
结果写入 dimensionalityReduction.txt
'''
def dimensionalityReduction(input,output):
    for readline in open(input):
        array = readline.strip('\n').split('$')
        compyAndGa1 = getMeta(array)
        vector1 = strTransVector(array)
        result = readyData(vector1,4,5)
        content = compyAndGa1 + ',' + str(result)
        print(content)
        saveSimfile(output, content)

if __name__ == "__main__":
    inputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\resultVector.txt"
    outputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\dimensionalityReduction.txt"
    outputPathTest = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\dimensionalityReduction_test.txt"
    # dimensionalityReduction(inputPath,outputPath)



    for readline in open(outputPath):
        array = readline.strip('\n')
        print(array.split(',')[1])



    # for readline in open(outputPath):
    #     array = readline.strip(']]').split(',')
    #     compyAndGa1 = getMeta(array)
    #     vector1 = strTransVector(array)
    #     print(compyAndGa1,vector1)
    #
    #     dict[] = readline.




