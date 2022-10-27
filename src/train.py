from cgitb import html
from sqlite3 import Timestamp
import nltk
import random
import string
import os
import email
from bs4 import BeautifulSoup
import json
import time
from email.parser import BytesParser, Parser
from email.policy import default

sampleRate = 1


def getSampleRate():
    return sampleRate


# 从文件生成数据集（dict）
def getSetFromFile(dataSet, file):
    lines = file.readlines()
    for line in lines:
        if line[-1] == "\n":
            line = line[:-1]
        if random.random() < sampleRate:
            label, path = line.split(" ")
            dataSet[path] = label


# 从 邮件全文str 解析出 邮件正文
def getText(lines):
    msg = email.message_from_string(lines)  # msg为Message类实例
    text = str()  # 存储正文
    for part in msg.walk():
        if not part.is_multipart():
            if part.get_content_type() == "text/html":
                content = part.get_payload()
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text()
                text += content
            elif part.get_content_type() == "text/plain":
                body = part.get_payload(
                    decode=True
                )  # to control automatic email-style MIME decoding (e.g., Base64, uuencode, quoted-printable)
                body = body.decode()
                text += body
    return text


# 将word添加到词典d中
def add(word, d):
    if word in d:
        d[word] += 1
    else:
        d[word] = 1


def getHeader(lines):
    msg = email.message_from_string(lines)

    headers = Parser(policy=default).parsestr(lines)
    date = msg.get("date")
    if date is None or len(date) == 0:
        date = "invalidTime"
    else:
        date = date.split(":")[0][-2:]
        if date[0] == " ":
            date = "0" + date[1]
        date = "sent_at_" + date

    sender = email.utils.parseaddr(msg.get("from"))[1].split("@")
    if len(sender) > 1:
        sender = sender[1]
    else:
        sender = "invalidSender"

    xmailer = headers["x-mailer"]
    if xmailer is None:
        xmailer = "invalidMailer"
    else:
        xmailer = xmailer.split(" ")[0]

    html = "notHtml"
    for part in email.message_from_string(lines).walk():
        if not part.is_multipart():
            if part.get_content_type() == "text/html":
                html = "hasHtml"
                break

    return [date, sender, xmailer, html]


# 在trainingSet上的第epoch轮训练
def train(trainingSet, epoch):
    print("*" * 20, "第%d轮训练开始" % epoch, "*" * 20)
    time.sleep(1)

    # 需要保存的数据
    spamCnt = 0  # 垃圾邮件总数
    hamCnt = 0  # 正常邮件总数
    spamWordCnt = 0  # 垃圾邮件中词语总数
    hamWordCnt = 0  # 正常邮件中词语总数
    spamWordDict = dict()  # 垃圾邮件中 "词语":频数
    hamWordDict = dict()  # 正常邮件中 "词语":频数

    # 额外特征
    spamHeader = dict()
    hamHeader = dict()

    for path in trainingSet:
        print("读取文件：" + path + "." * 6)
        label = trainingSet[path]
        if label == "spam":
            spamCnt += 1
        else:
            hamCnt += 1
        try:
            f = open(path, "r", encoding="utf-8")
            lines = f.read()  # 当前邮件全文字符串
            content = getText(lines)  # 当前邮件正文字符串
            content = content.lower()
            words = nltk.word_tokenize(content)  # 当前邮件词语列表
            for word in words:
                if word not in string.punctuation:
                    if label == "spam":
                        spamWordCnt += 1
                        add(word, spamWordDict)
                    else:
                        hamWordCnt += 1
                        add(word, hamWordDict)

            # 邮件header特征
            [date, sender, xmailer, html] = getHeader(lines)
            if label == "spam":
                add(date, spamHeader)
                add(sender, spamHeader)
                add(xmailer, spamHeader)
                add(html, spamHeader)
            else:
                add(date, hamHeader)
                add(sender, hamHeader)
                add(xmailer, hamHeader)
                add(html, hamHeader)

        except UnicodeDecodeError as e:
            print("剔除噪音文件: " + path)
        finally:
            f.close()
    # 保存训练得到的数据，供测试使用
    print("保存第%d轮训练的数据" % epoch + "." * 6)
    saveDir = "../trainingData/rate=" + str(sampleRate)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    spamWordDictF = open(saveDir + "/spamWordDict_" + str(epoch), "w", encoding="utf-8")
    hamWordDictF = open(saveDir + "/hamWordDict_" + str(epoch), "w", encoding="utf-8")
    cntF = open(saveDir + "/count_" + str(epoch), "w", encoding="utf-8")
    spamWordDict = dict(
        sorted(spamWordDict.items(), reverse=True, key=lambda item: item[1])
    )
    hamWordDict = dict(
        sorted(hamWordDict.items(), reverse=True, key=lambda item: item[1])
    )
    cntDict = {
        "spamCnt": spamCnt,
        "hamCnt": hamCnt,
        "spamWordCnt": spamWordCnt,
        "hamWordCnt": hamWordCnt,
    }
    spamWordDictF.write(json.dumps(spamWordDict))
    hamWordDictF.write(json.dumps(hamWordDict))
    cntF.write(json.dumps(cntDict))

    # 保存额外信息
    spamHeaderF = open(saveDir + "/spamHeader_" + str(epoch), "w", encoding="utf-8")
    hamHeaderF = open(saveDir + "/hamHeader_" + str(epoch), "w", encoding="utf-8")
    spamHeader = dict(
        sorted(spamHeader.items(), reverse=True, key=lambda item: item[1])
    )
    hamHeader = dict(sorted(hamHeader.items(), reverse=True, key=lambda item: item[1]))
    spamHeaderF.write(json.dumps(spamHeader))
    hamHeaderF.write(json.dumps(hamHeader))
    spamWordDictF.close()
    hamWordDictF.close()
    cntF.close()
    spamHeaderF.close()
    hamHeaderF.close()

    print("*" * 20, "第%d轮训练数据已全部保存。" % epoch, "*" * 20)

    # 保存testSet
    for i in range(5):
        fr = open("../dataFolds/fold" + str(i), "r", encoding="utf-8")
        fw = open(saveDir + "/testSet_" + str(i), "w", encoding="utf-8")
        fw.write(fr.read())
        fr.close()
        fw.close()


# 主函数
def fiveFoldTrain():
    # 创建文件夹保存训练过程中得到的数据
    if not os.path.exists("../trainingData"):
        os.makedirs("../trainingData")
    # 一共进行五轮
    for i in range(5):
        trainingSet = dict()
        # 每轮选择和i同编号的那个fold作test set，其余做training set
        for j in range(5):
            f = open("../dataFolds/fold" + str(j), "r")
            if j == i:
                pass
            else:
                getSetFromFile(trainingSet, f)
            f.close()
            # 至此第i轮的数据全部存在了训练集和测试集里
        """
        with open("b.json", "w") as f:
            js = json.dumps(trainingSet)
            f.write(js)
        with open("a.json", "w") as f:
            js = json.dumps(testSet)
            f.write(js)
        """
        train(trainingSet, i)
