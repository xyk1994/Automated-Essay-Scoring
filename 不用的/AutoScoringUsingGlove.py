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

csv_input = pd.read_csv(filepath_or_buffer="essay_set_1.tsv", delimiter="\t", engine="python", error_bad_lines=False)

# csv_input = pd.read_csv(filepath_or_buffer="Tests_Score_1031502.csv", encoding="ms932", sep=",")
csv_input = csv_input[["essay", "rater1_domain1"]]
uni = max(csv_input.loc[:, "rater1_domain1"].values) + 1
# uni = 3

#print(csv_input)
# l = list(range(len(csv_input)- 1))
# random.shuffle(l)

# use_train = sorted(l[0:300])
# use_test = sorted(l[301:])
data = csv_input
data2 = csv_input

# モデルクラスの定義
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size, drop_out, max):
        # クラスの初期化
        # :param vocab_size: 単語数
        # :param embed_size: 埋め込みベクトルサイズ
        # :param hidden_size: 隠れ層サイズ
        # :param out_size: 出力層サイズ
        # :param drop_out:ドロップアウト率

        super(LSTM_SentenceClassifier, self).__init__(
            # encode用のLink関数
            xe=L.EmbedID(vocab_size, embed_size, initialW=b, ignore_label=-1),
            eh=L.LSTM(embed_size, hidden_size),
            eh2=L.LSTM(embed_size, hidden_size),
            ii=L.Linear(hidden_size, 1),
            hh=L.Linear(hidden_size, hidden_size),
            # classifierのLink関数
            # hy = L.Linear(hidden_size * 2, out_size)
            hy=L.Linear(max * 2 - 2, out_size)
        )

    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x:　入力値
        # エンコード
        x = F.transpose_sequence(x)

        self.eh.reset_state()
        cel = []
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
            i = self.ii(h)
            cel.append(i)

        cel_back = []
        self.eh2.reset_state()
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh2(ee)
            i = self.ii(hh)
            cel_back.append(i)

        zz = F.concat((cel[0], cel_back[0]))

        for con in range(1, len(cel) - 1):
            kkk = F.concat((cel[con], cel_back[con]))
            zz = F.concat((zz, kkk))

        # 分類
        # z = F.concat((h,hh))
        len(zz)

        y = self.hy(zz)

        pp = F.softmax(y)
        # print(pp.data.argmax(axis=1))

        return y
'''
    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x:　入力値
        # エンコード
        x = F.transpose_sequence(x)


        self.eh.reset_state()
        cel = []
        for word in range(len(x)):
            e = []
            abc = x[word]
            for a in range(len(abc)):
                b = list(words.keys())[list(words.values()).index(abc[a])]
                b = words_vec[b]
                e.append(b)
            h = self.eh(e)
            i = self.ii(h)
            cel.append(i)

        cel_back = []
        self.eh2.reset_state()
        for word in range(1, len(x)):
            ee = []
            aaa = x[len(x) - word]
            for a in range(len(aaa)):
                bb = list(words.keys())[list(words.values()).index(aaa[a])]
                bb = words_vec[bb]
                ee.append(bb)
            #ee = x[len(x) - word]
            hh = self.eh2(ee)
            i = self.ii(hh)
            cel_back.append(i)

        zz = F.concat((cel[0], cel_back[0]))

        for con in range(1, len(cel) - 1):
            kkk = F.concat((cel[con], cel_back[con]))
            zz = F.concat((zz, kkk))

        # 分類
        # z = F.concat((h,hh))
        len(zz)

        y = self.hy(zz)

        pp = F.softmax(y)
        # print(pp.data.argmax(axis=1))

        return y
'''

# 学習
N = len(data)
#print(N)

data_x, data_t = [], []
train_x, train_t = [], []

for d in range(N):
    data_x.append(data.ix[d, 0])  # 文書
    data_t.append(data.ix[d, 1])  # ラベル

for d in range(N):
    train_x.append(data.ix[d, 0])  # 文書
    train_t.append(data.ix[d, 1])  # ラベル


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



# 単語辞書
def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return [dict[key] for key in keys]

words_vec = {}
f = open('glove.6B.100d.txt', 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    words_vec[word] = coefs
f.close()

a = []
b = []
c = []
for i in words_vec.keys():
    a.append([i, words_vec[i]])
    b.append(words_vec[i])
    c.append([i])
b = np.array(b)

words = {}
for i in range(len(c)):
    words[c[i][0]] = len(words)



# 文章を単語ID配列にする
data_x_vec = []
train_x_vec = []

for sentence in data_x:
    sentence_words = sentence2words(sentence)
    # backの時に用いる
    # sentence_words.reverse()
    sentence_ids = []
    for word in sentence_words:
        if words.get(word, None) is None:
            sentence_ids.append(words['<unk>'])
        else:
            sentence_ids.append(words[word])
    data_x_vec.append(sentence_ids)

for sentence in train_x:
    sentence_words = sentence2words(sentence)
    # backの時に用いる
    # sentence_words.reverse()
    sentence_ids = []
    for word in sentence_words:
        if words.get(word, None) is None:
            sentence_ids.append(words['<unk>'])
        else:
            sentence_ids.append(words[word])
    train_x_vec.append(sentence_ids)

# 文章の長さを揃えるため、-1パディングする（系列を覚えておきやすくするため、前パディングする）
max_sentence_size = 0
for sentence_vec in data_x_vec:
    if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)
for sentence_vec in train_x_vec:
    if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)

for sentence_ids in data_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)  # 先頭に追加
for sentence_ids in train_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)  # 先頭に追加

# データセット
#data_x_vec = np.array(data_x_vec, dtype="int32")
#train_x_vec = np.array(train_x_vec, dtype="int32")

data_t = np.array(data_t, dtype="int32")  # label
train_t = np.array(train_t, dtype="int32")  # label

dataset = []
for x, t in zip(data_x_vec, data_t):
    dataset.append((x, t))

# 定数
EPOCH_NUM = 10
EMBED_SIZE = 100
HIDDEN_SIZE = 200
BATCH_SIZE = 10
OUT_SIZE = uni

# モデルの定義
model = L.Classifier(LSTM_SentenceClassifier(
    vocab_size=len(words),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    out_size=OUT_SIZE,
    drop_out=0.5,
    max=max_sentence_size
))
optimizer = optimizers.Adam()
optimizer.setup(model)

#print(dataset)

# 学習開始
train, test = chainer.datasets.split_dataset_random(dataset, N - 350)
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport(
    ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy",
     "elapsed_time"], ))  # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.ProgressBar())  # プログレスバー出力
trainer.run()

# chainer.serializers.save_npz("mymodel.npz", model)

# test
xt = Variable(train_x_vec)
yt = model.predictor(xt)
ans = yt.data
ok = 0
for i in range(len(train_t)):
    if np.argmax(ans[i, :]) == train_t[i]:
        ok += 1
    if i == 0:
        df = pd.DataFrame([[np.argmax(ans[i, :]), train_t[i]]], columns=['predict', "true"])
    if i != 0:
        df2 = pd.DataFrame([[np.argmax(ans[i, :]), train_t[i]]], columns=['predict', "true"])
        df = df.append(df2)

# df.to_csv("test.csv")

print((ok * 1.0) / len(train_t))
