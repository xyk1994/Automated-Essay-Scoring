import re
import csv
import pickle
import numpy as np
import chainer
from chainer import Chain, optimizers, training, Variable
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import pandas as pd
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from chainer.training import triggers
from chainer import serializers
import random
from operator import itemgetter
import codecs
import fileinput
import sys
from matplotlib import pyplot as plt

csv_input = pd.read_csv(filepath_or_buffer="essayset1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)

csv_input = csv_input[["essay", "rater1_domain1", "word_length"]]

data = csv_input
data = np.random.permutation(data)
save_data = pd.DataFrame(data)
save_data.to_csv('save_data.csv', sep='\t')

print("AES_with_word_length")

# モデルクラスの定義
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, drop_out, max):
        # クラスの初期化
        # :param vocab_size: 単語数
        # :param embed_size: 埋め込みベクトルサイズ
        # :param hidden_size: 隠れ層サイズ
        # :param out_size: 出力層サイズ
        # :param drop_out:ドロップアウト率

        super(LSTM_SentenceClassifier, self).__init__(
            # encode用のLink関数
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.LSTM(embed_size, hidden_size),
            eh2=L.LSTM(embed_size, hidden_size),
            ii=L.Linear(hidden_size, 1),
            hh=L.Linear(hidden_size, hidden_size),
            # classifierのLink関数
            # hy = L.Linear(hidden_size * 2, out_size)
            hy=L.Linear(max * 2 - 2 + 1, 1),

        )

    def __call__(self, x, Label, feature):
        # 順伝播の計算を行う関数
        # :param x:　入力値
        # :param y:  label
        # エンコード
        x = F.transpose_sequence(x)

        self.eh.reset_state()
        cel = []
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
            i = self.ii(h)
            cel.append(i)
        # print("cel = ", cel)

        cel_back = []
        self.eh2.reset_state()
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh2(ee)
            i = self.ii(hh)
            cel_back.append(i)
        # print("cel_back = ", cel)

        zz = F.concat((cel[0], cel_back[0]))

        # print("len(zz1) = ", len(zz))
        # print("zz1 = ",len(zz[0]))

        for con in range(1, len(cel) - 1):
            kkk = F.concat((cel[con], cel_back[con]))
            zz = F.concat((zz, kkk))
        # print(zz)
        # print("len(zz2) = ", len(zz))
        # print("zz2 = ",len(zz[0]))
        zzz = F.concat((zz, feature), axis=1)
        # 分類
        # z = F.concat((h,hh))

        y = self.hy(zzz)

        # pp = F.softmax(y)
        # print(pp.data.argmax(axis=1))
        #score = F.sigmoid(y) * 6
        # print("y = ", y)
        # print("new_y", new_y)
        # print("score = ", score)

        # loss = 1 / 2 * ((score - yl) ** 2)

        rmse = 0
        for i in range(BATCH_SIZE):
            rmse += ((y[i] - Label[i]) ** 2)
        rmse = F.sqrt(rmse / BATCH_SIZE)
        print("RMSE = ", rmse.data)
        return rmse, y
