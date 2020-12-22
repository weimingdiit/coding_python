import nltk
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

from Tools import transCsvTotxtArray, transCsvTotxtMap, transCsvToCompanyName, savefile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

"""
参考：

https://www.52nlp.cn/%E5%A6%82%E4%BD%95%E8%AE%A1%E7%AE%97%E4%B8%A4%E4%B8%AA%E6%96%87%E6%A1%A3%E7%9A%84%E7%9B%B8%E4%BC%BC%E5%BA%A6%E4%BA%8C
"""


def getSimilaritiesD1(texts, keyword):
    # 将文本集合生成分词列表--针对中文的采用jieba分词
    # 英文采用nltk分词器进行分词
    texts = [word_tokenize(text) for text in texts]
    # 2、基于文本集建立【词典】，并获得词典特征数
    dictionary = Dictionary(texts)
    num_features = len(dictionary.token2id)
    # 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
    kw_vector = dictionary.doc2bow(lcut(keyword))
    # 4、创建【TF-IDF模型】，传入【语料库】来训练
    tfidf = TfidfModel(corpus)
    # 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
    tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
    tf_kw = tfidf[kw_vector]
    # 6、相似度计算
    sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
    similarities = sparse_matrix.get_similarities(tf_kw)
    for e, s in enumerate(similarities, 1):
        print('kw 与 text%d 相似度为：%.2f' % (e, s))


"""
分词，并且去掉停用词
"""


def getSplit(data):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(data)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w.lower())
    return wordsFiltered


def getSimilaritiesD2(documents, keyword):
    texts = [getSplit(document) for document in documents]
    # print(texts)
    # 文档抽象出词袋
    dictionary = Dictionary(texts)
    # print(dictionary)
    # print(dictionary.token2id)
    # 用字符串表示的文档转换为用id表示的文档向量
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(corpus)

    # 基于这些“训练文档”计算一个TF-IDF“模型”
    tfidf = TfidfModel(corpus)
    # 用词频表示文档向量 转化为  用tf-idf值表示的文档向量
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print(doc)

    # 训练一个LSI模型
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    lsi.print_topics(2)
    corpus_lsi = lsi[corpus_tfidf]
    # for doc in corpus_lsi:
    #     print(doc)

    # 创建索引
    index = similarities.MatrixSimilarity(lsi[corpus])

    # 用之前训练好的LSI模型将其映射到二维的topic空间
    kw_bow = dictionary.doc2bow(getSplit(keyword))
    kw_lsi = lsi[kw_bow]
    # print(kw_lsi)

    # 计算keyword和index中doc的余弦相似度了
    sims = index[kw_lsi]
    # 按照相似度排序
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(sort_sims)
    return sort_sims


def saveResultToTxt(csv_path, sava_path):
    # 获取文本编号和企业的映射关系
    companyWithNo = transCsvToCompanyName(csv_path)

    # 计算相似度
    texts = transCsvTotxtArray(csv_path)
    with open(sava_path, "wb") as fp:
        for index, text in enumerate(texts):
            results = getSimilaritiesD2(texts, text)
            for result in results:
                finalResult = companyWithNo[index] + "," + str(index) + "," + companyWithNo[result[0]] + "," + str(
                    result[0]) + "," + str(result[1]) + "\n"
                fp.write(finalResult.encode('utf-8'))
                print(finalResult)


if __name__ == "__main__":
    # 文本集和搜索词
    # texts = ['吃鸡这里所谓的吃鸡并不是真的吃鸡，也不是谐音词刺激的意思',
    #          '而是出自策略射击游戏《绝地求生：大逃杀》里的台词',
    #          '我吃鸡翅，你吃鸡腿']
    # keyword = '玩过吃鸡？今晚一起吃鸡'
    #
    # getSimilaritiesD1(texts,keyword)

    csv_path1 = "E:\\project\\xmyselfProject\\data\\patent_data\\train_patent\\qxwith11company.csv"
    sava_path1 = "E:\\project\\xmyselfProject\\data\\patent_data\\train_patent\\qx_result.txt"
    saveResultToTxt(csv_path1, sava_path1)
