from functools import total_ordering
import math
import nltk
import train
import os
import numpy as np
import json
import re


def smooth(part, total, alpha=1e-6, M=2):
    return (part + alpha) / (total + alpha * M)


def fiveFoldTest():
    train.fiveFoldTrain()
    rate = train.getSampleRate()

    if not os.path.exists("../res"):
        os.makedirs("../res")
    resPath = "../res/rate=" + str(rate)
    if not os.path.exists(resPath):
        os.makedirs(resPath)
    fw = open(resPath + "/result.txt", "w", encoding="utf-8")
    fw.write("epoch\taccuracy\tprecision\trecall  \tF1\n")

    evaluation = np.array([0, 0, 0, 0])  # 存储最终5轮数据

    for i in range(5):
        # 导入训练数据
        saveDir = "../trainingData/rate=" + str(rate)
        spamWordDictF = open(saveDir + "/spamWordDict_" + str(i), "r", encoding="utf-8")
        hamWordDictF = open(saveDir + "/hamWordDict_" + str(i), "r", encoding="utf-8")
        cntF = open(saveDir + "/count_" + str(i), "r", encoding="utf-8")

        spamWordDict = json.loads(spamWordDictF.read())
        hamWordDict = json.loads(hamWordDictF.read())
        cntDict = json.loads(cntF.read())

        spamCnt = cntDict["spamCnt"]
        hamCnt = cntDict["hamCnt"]
        spamWordCnt = cntDict["spamWordCnt"]
        hamWordCnt = cntDict["hamWordCnt"]

        spamWordDictF.close()
        hamWordDictF.close()
        cntF.close()

        # 导入额外特征数据
        spamHeaderF = open(saveDir + "/spamHeader_" + str(i), "r", encoding="utf-8")
        hamHeaderF = open(saveDir + "/hamHeader_" + str(i), "r", encoding="utf-8")
        spamHeader = json.loads(spamHeaderF.read())
        hamHeader = json.loads(hamHeaderF.read())

        # 生成测试集dict
        testSetF = open(saveDir + "/testSet_" + str(i), "r", encoding="utf-8")
        testSet = dict()
        lines = testSetF.readlines()
        for line in lines:
            if line[-1] == "\n":
                line = line[:-1]
            label, path = line.split(" ")
            testSet[path] = label
        testSetF.close()

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for path in testSet:
            label = testSet[path]
            try:
                f = open(path, "r", encoding="utf-8")
                lines = f.read()
                content = train.getText(lines)
                content = content.lower()
                content = re.sub(r"[^\w\s]", "", content)
                words = nltk.word_tokenize(content)

                # 邮件头数据
                headers = train.getHeader(lines)

                # 朴素贝叶斯分类器计算公式中的第一项
                spamP = math.log(spamCnt / (spamCnt + hamCnt))
                hamP = math.log(hamCnt / (hamCnt + spamCnt))

                # 朴素贝叶斯分类器计算公式中的其他项
                for word in words:
                    if word in spamWordDict:
                        spamP += math.log(smooth(spamWordDict[word], spamWordCnt))
                    else:
                        spamP += math.log(smooth(0, spamWordCnt))

                    if word in hamWordDict:
                        hamP += math.log(smooth(hamWordDict[word], hamWordCnt))
                    else:
                        hamP += math.log(smooth(0, hamWordCnt))

                # 计算额外特征
                w = 10  # 权重
                for header in headers:
                    if header in spamHeader:
                        spamP += w * math.log(smooth(spamHeader[header], spamCnt))
                    else:
                        spamP += w * math.log(smooth(0, spamCnt))
                    if header in hamHeader:
                        hamP += w * math.log(smooth(hamHeader[header], hamCnt))
                    else:
                        hamP += w * math.log(smooth(0, hamCnt))

                # 将垃圾邮件作为正类
                if spamP >= hamP:
                    res = "spam"
                    if label == "spam":
                        TP += 1
                    else:
                        FP += 1
                else:
                    res = "ham"
                    if label == "ham":
                        TN += 1
                    else:
                        FN += 1

            except UnicodeDecodeError as e:
                continue
            finally:
                f.close()
        tt = TP + TN + FP + FN
        accuracy = (TP + TN) / tt
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

        fw.write("%d\t%f\t%f\t%f\t%f\n" % (i, accuracy, precision, recall, F1))
        evaluation = np.add(evaluation, np.array([accuracy, precision, recall, F1]))

    evaluation = np.divide(evaluation, 5)
    fw.write(
        "average\t%f\t%f\t%f\t%f\n"
        % (evaluation[0], evaluation[1], evaluation[2], evaluation[3])
    )
    fw.close()


fiveFoldTest()
