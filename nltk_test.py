# from janome.tokenizer import Tokenizer
import re
import sys
import numpy as np
import chainer
from chainer import Chain, optimizers, training, Variable, cuda
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import pandas as pd
import math
import random
#import backtrace
#from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from chainer.functions.loss.mean_squared_error import mean_squared_error
from chainer.training import extensions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)
csv_input = csv_input[["essay"]]
# data = csv_input
# a = data.ix[0, 0]
# print(type(a))

data = csv_input.values.tolist()
data = [x[0] for x in data]
# print(data)
data_tokenize = []
stopWords = set(stopwords.words('english'))
for i in range(len(data)):
    wordsFiltered = []
    words = word_tokenize(data[i])
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    data_tokenize.append(wordsFiltered)
data_tokenize = np.array(data_tokenize)
# print(data_tokenize)
print(data_tokenize.shape)
'''
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(words)
'''
max = 0
for eassy in data_tokenize:
    if max < len(eassy):
        max = len(eassy)
print(max)