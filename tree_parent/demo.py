from Tools import readCsv, saveSimfile
from tree_parent.SimTree import cos_sim


def createM(list):
    currLineListFloat = []
    for i in list:
        currLineListFloat.append(int(i))
    return currLineListFloat




if __name__ == "__main__":
    inputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\CSV\\sample.csv"
    outputPath = "E:\\myselfFile\\zbbFile\\矩阵\\SimTree\\CSV\\sample_result.txt"
    dict = {}

    csv_rows = readCsv(inputPath)
    for row in csv_rows:
        key = row.pop(0)
        value = createM(row)
        dict[key] = value

    for (k1, v1) in dict.items():
        for (k2, v2) in dict.items():
            k =  k1 + '#' +k2
            v = cos_sim(v1,v2)
            print(k,v)
            content = k + ',' + str(v)
            saveSimfile(outputPath, content)

