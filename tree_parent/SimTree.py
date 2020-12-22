
import numpy as np
import datetime
from sklearn.decomposition import PCA
from Tools import savefile, saveSimfile



# # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# X = np.array([123412,13234,2345,345,5,345,46,3456,357,7567,45,74,9,3,7])
# Y = X.reshape(-1,1)
# pca = PCA(n_components = 1)
# #等价于pca.fit(X) pca.transform(X)
# newX = pca.fit_transform(Y)
# #将降维后的数据转换成原始数据
# # invX = pca.inverse_transform(X)
# print(X)
# print(newX)
# # print(invX)
# print(pca.explained_variance_ratio_)



def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return str(sim)


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
            # print(compyAndGa1,compyAndGa2,cos_sim(vector1,vector2))

            content = compyAndGa1 + ',' + compyAndGa2 + ',' + cos_sim(vector1, vector2)
            saveSimfile("E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\finalSimResult.txt", content)

