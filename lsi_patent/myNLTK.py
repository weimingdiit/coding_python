import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Tools import savefile, readfile, readCsv


def getSegment(txt_path, seg_path):
    nltk.download('stopwords')
    data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(data)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    print(wordsFiltered)


def getTfIdf():
    print("ajsdjba")


if __name__ == "__main__":
    # 将csv文件的每一行数据保存为单独的txt文件
    csv_path = "E:\\project\\xmyselfProject\\data\\patent_data\\train_patent\\weimingtest.csv"
    txt_path = "E:\\project\\xmyselfProject\\chinese_text_classification\\lsi_patent\\train_txt\\"
    seg_path = "E:\\project\\xmyselfProject\\chinese_text_classification\\lsi_patent\\train_seg\\"
    # transCsvTotxt(csv_path,txt_path)
    getSegment(txt_path, seg_path)
    getTfIdf()
