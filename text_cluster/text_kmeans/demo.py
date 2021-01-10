# !--encoding=utf-8

from __future__ import print_function

from jieba import xrange
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans

'''
优点：
1、原理比较简单，实现也是很容易，收敛速度快。
2、当结果簇是密集的，而簇与簇之间区别明显时, 它的效果较好。
3、主要需要调参的参数仅仅是簇数k。
缺点：
1、K值需要预先给定，很多情况下K值的估计是非常困难的。
2、K-Means算法对初始选取的质心点是敏感的，不同的随机种子点得到的聚类结果完全不同 ，对结果影响很大。
3、对噪音和异常点比较的敏感。用来检测异常值。
4、采用迭代方法，可能只能得到局部的最优解，而无法得到全局的最优解。
'''
dataset = []


def loadDataset():
    '''导入文本数据集'''
    f = open('E:\\myselfFile\\zbbFile\\矩阵\\cluster_julei\\聚类样本.txt', 'r')
    # dataset = []
    for line in f.readlines():
        abstract = line.split(',')[1].strip()
        dataset.append(abstract)
    f.close()
    return dataset


def transform(dataset, n_features=1000):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, use_idf=True, analyzer='word',
                                 stop_words='english')
    X = vectorizer.fit_transform(dataset)
    return X, vectorizer


'''
param:
lableNum 标识每一聚类结果的标签（关键词）
showClusterPatent 标识是否展示每个文本对应的分类类别（默认不展示，false）
'''


def train(X, vectorizer, true_k, minibatch=False, showLable=True, lableNum=6, showClusterPatent=False):
    # 使用采样数据还是原始数据训练k-means，
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=False)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=5,
                    verbose=False)
    km.fit(X)
    # 每个点，到簇类中心的距离之和，用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print('=====================', 'true_k = ' + str(true_k))
    print('all distance:' + str(km.inertia_))
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :lableNum]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    if showClusterPatent:
        for i in range(len(result)):
            print('类别：' + str(result[i]), str(dataset[i]))

    print('Cluster distribution:')
    print(dict([(i, result.count(i)) for i in result]))
    return -km.score(X)


def test():
    '''测试选择最优参数'''

    # dataset = loadDataset()
    print("%d documents" % len(dataset))
    X, vectorizer = transform(dataset, n_features=1000)
    true_ks = []
    scores = []
    for i in xrange(3, 20, 1):
        score = train(X, vectorizer, true_k=i) / len(dataset)
        print(i, score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(true_ks, scores, label="error", color="red", linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()


'''
param：
true_k 代表文本需要被聚成多少类
'''


def out(true_k):
    '''在最优参数下输出聚类结果'''
    # dataset = loadDataset()
    X, vectorizer = transform(dataset, n_features=1000)
    score = train(X, vectorizer, true_k, showLable=True) / len(dataset)
    print(score)


if __name__ == "__main__":
    loadDataset()
    # test()
    out(5)
