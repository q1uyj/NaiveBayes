from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

x = [0.05, 0.5, 1]
y = [0.899834, 0.910504, 0.904870]
ymin = [0.870588, 0.894573, 0.899875]
ymax = [0.939580, 0.952674, 0.909176]
plt.plot(x, y, color="r", label="average")
plt.plot(x, ymin, color="g", label="min")
plt.plot(x, ymax, color="b", label="max")
plt.title("influence of size of training set")
plt.xticks([0.05, 0.5, 1])
plt.legend()
plt.show()
