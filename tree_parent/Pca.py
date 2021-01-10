import datetime

import numpy as np
from sklearn.decomposition import PCA
import sys

# returns choosing how many main factors
from Tools import saveSimfile
from tree_parent.SimTree import getMeta, strTransVector


def fun0():
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = np.array([123412, 13234, 2345, 345, 5, 345, 46, 3456, 357, 7567, 45, 74, 9, 3, 7])
    Y = X.reshape(3, 5)

    pca = PCA(n_components=2)
    # 等价于pca.fit(X) pca.transform(X)
    pca.fit(Y)
    newX = pca.fit_transform(Y)
    # 将降维后的数据转换成原始数据
    # invX = pca.inverse_transform(X)

    print(X)
    print(Y)
    print(newX)
    print(pca.explained_variance_ratio_)


def index_lst(lst, component=0, rate=0):
    # component: numbers of main factors
    # rate: rate of sum(main factors)/sum(all factors)
    # rate range suggest: (0.8,1)
    # if you choose rate parameter, return index = 0 or less than len(lst)
    if component and rate:
        print('Component and rate must choose only one!')
        sys.exit(0)
    if not component and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif component:
        print('Choosing by component, components are %s......' % component)
        return component
    else:
        print('Choosing by rate, rate is %s ......' % rate)
        for i in range(1, len(lst)):
            if sum(lst[:i]) / sum(lst) >= rate:
                return i
        return 0


def fun1(Mat, cov_Mat, n):
    # PCA by original algorithm
    # eigvalues and eigenvectors of covariance Matrix with eigvalues descending
    U, V = np.linalg.eigh(cov_Mat)
    # Rearrange the eigenvectors and eigenvalues
    U = U[::-1]
    for i in range(n):
        V[i, :] = V[i, :][::-1]
    # choose eigenvalue by component or rate, not both of them euqal to 0
    Index = index_lst(U, component=2)  # choose how many main factors
    if Index:
        v = V[:, :Index]  # subset of Unitary matrix
    else:  # improper rate choice may return Index=0
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i]) / sum(U) for i in range(1, len(U) + 1)])
        sys.exit(0)
    # data transformation
    T1 = np.dot(Mat, v)
    # print the transformed data
    print('We choose %d main factors.' % Index)
    print('After PCA transformation, data becomes:\n', T1)


def fun2(Mat, cov_Mat):
    # PCA by original algorithm using SVD
    print('\nMethod 2: PCA by original algorithm using SVD:')
    # u: Unitary matrix,  eigenvectors in columns
    # d: list of the singular values, sorted in descending order
    u, d, v = np.linalg.svd(cov_Mat)
    Index = index_lst(d, rate=0.95)  # choose how many main factors
    T2 = np.dot(Mat, u[:, :Index])  # transformed data
    print('We choose %d main factors.' % Index)
    print('After PCA transformation, data becomes:\n', T2)


def fun3(mat, n):
    # PCA by Scikit-learn
    pca = PCA(n_components=n)  # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    # print('\nMethod 3: PCA by Scikit-learn:')
    # print('After PCA transformation, data becomes:')
    result = pca.fit_transform(mat)  # transformed data
    # print(result)
    return result

def fun4(mat, n):
    # 83923 = 7*19*631
    X = np.array(mat)
    Y = X.reshape(19, 631*7)

    pca = PCA(n_components=n)
    # 等价于pca.fit(X) pca.transform(X)
    pca.fit(Y)
    newX = pca.fit_transform(Y)
    return newX
    # 将降维后的数据转换成原始数据
    # invX = pca.inverse_transform(X)
    # print(newX)
    # print(pca.explained_variance_ratio_)



