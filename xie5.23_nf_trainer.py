import re
import csv
import pickle
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import pandas as pd
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from chainer.training import triggers
import random
from operator import itemgetter
import codecs
import fileinput
import sys
from matplotlib import pyplot as plt
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions

csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)

csv_input = csv_input[["essay", "rater1_domain1"]]
data = csv_input
print("Automated Eassy Scoring")

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
            hy=L.Linear(max * 2 - 2, 1),

        )

    # Forward propagation
    def __call__(self, x):
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
        #zzz = F.concat((zz, feature), axis=1)
        # 分類
        # z = F.concat((h,hh))

        y = self.hy(zz)

        # pp = F.softmax(y)
        # print(pp.data.argmax(axis=1))
        #score = F.sigmoid(y) * 6
        # print("y = ", y)
        # print("new_y", new_y)
        # print("score = ", score)

        # loss = 1 / 2 * ((score - yl) ** 2)
        '''
        rmse = 0
        for i in range(BATCH_SIZE):
            rmse += ((y[i] - Label[i]) ** 2)
        rmse = F.sqrt(rmse / BATCH_SIZE)
        print("RMSE = ", rmse.data)
        '''
        return y


def lossfun(predict, true):
    r = 0
    for i in range(BATCH_SIZE):
        r += ((predict[i] - true[i]) ** 2)
    rmse = F.sqrt(r / BATCH_SIZE)
    return rmse


def sentence2words(sentence):
    stopwords = ["i", "a", "an", "the", "and", "or", "if", "is", "are", "am", "it", "this", "that", "of", "from", "in",
                 "on"],
    sentence = sentence.lower()  # 小文字化
    sentence = sentence.replace("\n", "")  # 改行削除
    sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~],"), " ", sentence)  # 記号をスペースに置き換え
    sentence = sentence.split(" ")  # スペースで区切る
    sentence_words = []
    for word in sentence:
        # if (re.compile(r"^.*[0-9],+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
        # continue
        if word in stopwords:  # ストップワードに含まれるものは除外
            continue
        sentence_words.append(word)
    return sentence_words


'''
# 交差検証法
def CrossValidation(dataset, n_fold, order):
    whole_size = len(dataset)
    if whole_size % n_fold == 0:
        CVsize = whole_size // n_fold
    else:
        CVsize = (whole_size // n_fold) + 1

    if order == 0:
        test = dataset[0: CVsize]
        train = dataset[CVsize: len(dataset)]
    elif order == n_fold - 1:
        test = dataset[order * CVsize: len(dataset)]
        train = dataset[0: order * CVsize]
    else:
        test = dataset[order * CVsize: (order + 1) * CVsize]
        train = np.concatenate((dataset[0: order * CVsize], dataset[(order + 1) * CVsize: len(dataset)]), axis=0)
    return train, test
'''

data_t, data_l = [], []
N = len(data)

for i in range(N):
    data_t.append(data.ix[i, 0])
    data_l.append(data.ix[i, 1])

# 単語辞書
words = {}
for sentence in data_t:
    sentence_words = sentence2words(sentence)
    for word in sentence_words:
        if word not in words:
            words[word] = len(words)

# 文章を単語ID配列にする
data_t_vec = []
for sentence in data_t:
    sentence_words = sentence2words(sentence)
    # backの時に用いる
    # sentence_words.reverse()
    sentence_ids = []
    for word in sentence_words:
        sentence_ids.append(words[word])
    data_t_vec.append(sentence_ids)

# 文章の長さを揃えるため、-1パディングする（系列を覚えておきやすくするため、前パディングする）
max_sentence_size = 0
for sentence_vec in data_t_vec:
    if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)
print("max_sentence_size = ", max_sentence_size)

for sentence_ids in data_t_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)  # 先頭に追加


# 定数
EPOCH_NUM = 10
EMBED_SIZE = 200
HIDDEN_SIZE = 200
BATCH_SIZE = 10
losses_train = []
losses_test = []
predict = []
n_fold = 5

# モデルの定義
model = LSTM_SentenceClassifier(vocab_size=len(words),
                                embed_size=EMBED_SIZE,
                                hidden_size=HIDDEN_SIZE,
                                drop_out=0.5,
                                max=max_sentence_size
                                )

# データセット
# dataset = []
data_t_vec = np.array(data_t_vec, dtype="int32")  # text
data_l = np.array(data_l, dtype="int32")  # label
data_l = data_l.reshape(len(data_l), 1)
# print(data_t_vec.shape)
# print(data_l.shape)
dataset = list(zip(data_t_vec, data_l))

index = chainer.datasets.sub_dataset.get_cross_validation_datasets(dataset, n_fold, order=None)
fold = 0  # n_fold = 5,fold = 0,1,2,3,4
index_loop = index[fold]
index_train = index_loop[0]
index_test = index_loop[1]

# Iterator
train_iter = iterators.SerialIterator(index_train, BATCH_SIZE)
test_iter = iterators.SerialIterator(index_test, BATCH_SIZE, repeat=False, shuffle=False)

# モデルの調整
model = L.Classifier(model, lossfun=lossfun)
model.compute_accuracy = False

# GPUの設定(0:GPU / -1:CPU)
gpu_id = -1
if gpu_id >= 0:
    model.to_gpu(gpu_id)

# Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Updater
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Trainer
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")

# 検証データで評価
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))

# Output
# trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"],)) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time'],))
trainer.extend(extensions.ProgressBar())  # プログレスバー出力

# 学習の実行
trainer.run()