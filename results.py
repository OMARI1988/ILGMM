import colorsys
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import numpy as np
import pickle

A = {'and': 0.2589578826392645, 'right': 0.2663213179040983, 'Purple': 0.44999999999999996, 'is': 0.25865227261521712, 'Fuchsia': 0.75095238095238104, 'yellow': 0.80138453217955674, 'navy': 0.5, 'close': 0.26963470745145551, 'lime': 0.79000000000000004, 'blue': 0.77000000000000002, 'from': 0.26236995370774746, 'near': 0.26951168700350275, 'to': 0.26005028628792387, 'black': 0.72262626262626262, 'above': 0.28637880491084217, 'white': 0.64972335600907039, 'red': 0.76276296296296286, 'object': 0.25050075100125152, 'bottom_right': 0.2637493663073906, 'far': 0.26876012500000002, 'top_right': 0.26843130772113127, 'maroon': 0.54000000000000004, 'very_far': 0.2709047816170323, 'olive': 0.55000000000000004, 'bottom_left': 0.29476306708189548, 'silver': 0.29787401574803152, 'a': 0.23747918834547344, 'gray': 0.24880968778696047, 'top_left': 0.27205663743124719, 'this': 0.17304378782484872, 'of': 0.2589578826392645, 'aqua': 0.80000000000000004, 'below': 0.27314744801512292, 'green': 0.56000000000000005, 'teal': 0.52000000000000002, 'the': 0.2554326068180664, 'left': 0.27996975425330811}

words = sorted(A, key=A.__getitem__)

values = []
for i in words:
	values.append(A[i])

print words,values

fig = plt.figure()
ax = plt.subplot(111)
ax.bar(range(len(words)), values, width=.75)
width=.75
ax.set_xticks(np.arange(len(words)) + width/2)
ax.set_xticklabels(words, rotation=90)
plt.show()

