import random
import os
import time

"""
 - Todo
将所有的样本随机均分成5份，保存在5个文件里
"""


def main():
    with open("../trec06p/label/index", "r") as f:
        lines = f.readlines()

    if not os.path.exists("../dataFolds"):
        os.makedirs("../dataFolds")

    setters = []
    for i in range(5):
        setters.append(open("../dataFolds/fold" + str(i), "w"))

    for line in lines:
        # print(line)
        # time.sleep(1)
        label, path = line.split(" ")
        path = "../trec06p" + path[2:]
        newline = label + " " + path
        # print(newline)
        # time.sleep(1)
        index = random.randint(0, 4)
        setters[index].writelines(newline)

    for i in range(5):
        setters[i].close()


if __name__ == "__main__":
    main()
