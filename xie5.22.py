# coding:utf-8
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


csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature(300).tsv", delimiter="\t", engine="python", error_bad_lines=False)

csv_input = csv_input[["essay", "rater1_domain1", "word_length"]]
# uni = max(csv_input.loc[:, "rater1_domain1"].values) + 1

print("AES_with_word_length")
# print(csv_input)
# l = list(range(len(csv_input)- 1))
# random.shuffle(l)

# use_train = sorted(l[0:300])
# use_test = sorted(l[301:])
data = csv_input
# data2 = csv_input


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


# 学習
N = len(data)
# print(N)

data_x, data_t, data_f = [], [], []  # x = text, t = label, f = feature
train_x, train_t, train_f = [], [], []


for d in range(N):
    data_x.append(data.ix[d, 0])  # 文書
    data_t.append(data.ix[d, 1])  # ラベル
    data_f.append(data.ix[d, 2])  # 特徴量


for d in range(N):
    train_x.append(data.ix[d, 0])  # 文書
    train_t.append(data.ix[d, 1])  # ラベル
    train_f.append(data.ix[d, 2])  # 特徴量

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
words = {}
for sentence in data_x:
    sentence_words = sentence2words(sentence)
    for word in sentence_words:
        if word not in words:
            words[word] = len(words)

for sentence in train_x:
    sentence_words = sentence2words(sentence)
    for word in sentence_words:
        if word not in words:
            words[word] = len(words)

# 文章を単語ID配列にする
data_x_vec = []
train_x_vec = []

for sentence in data_x:
    sentence_words = sentence2words(sentence)
    # backの時に用いる
    # sentence_words.reverse()
    sentence_ids = []
    for word in sentence_words:
        sentence_ids.append(words[word])
    data_x_vec.append(sentence_ids)

for sentence in train_x:
    sentence_words = sentence2words(sentence)
    # backの時に用いる
    # sentence_words.reverse()
    sentence_ids = []
    for word in sentence_words:
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
print("max_sentence_size = ", max_sentence_size)

for sentence_ids in data_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)  # 先頭に追加
for sentence_ids in train_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)  # 先頭に追加

# データセット
data_x_vec = np.array(data_x_vec, dtype="int32")
train_x_vec = np.array(train_x_vec, dtype="int32")

data_t = np.array(data_t, dtype="int32")  # label
train_t = np.array(train_t, dtype="int32")  # label

data_f = np.array(data_f, dtype="int32")
train_f = np.array(train_f, dtype="int32")

dataset = []
for x, t, f in zip(data_x_vec, data_t, data_f):
    dataset.append((x, t, f))

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


# 定数
EPOCH_NUM = 50
EMBED_SIZE = 200
HIDDEN_SIZE = 200
BATCH_SIZE = 10
losses_train = []
losses_test = []
predict = []
n_fold = 5
OUT_SIZE = 1

# モデルの定義
model = LSTM_SentenceClassifier(vocab_size=len(words),
                                embed_size=EMBED_SIZE,
                                hidden_size=HIDDEN_SIZE,
                                drop_out=0.5,
                                max=max_sentence_size
                                )
optimizer = optimizers.Adam()
optimizer.setup(model)

# index = np.random.permutation(dataset)
# print(type(index))
# np.savetxt('index.csv', index, delimiter='\t')
# save_data = pd.DataFrame(index)
# save_data.to_csv('index.csv')

#index_train, index_test = CrossValidation(index, 5, order=0) # order = 0,1,2,3,4
index = chainer.datasets.sub_dataset.get_cross_validation_datasets(dataset, n_fold, order=None)
# for fold in range(n_fold):

fold = 4  # n_fold = 5,fold = 0,1,2,3,4
index_loop = index[fold]
index_train = index_loop[0]
index_test = index_loop[1]
#print(len(index_train))

for j in range(EPOCH_NUM):
    # index = np.random.permutation(dataset)
    # index_test, index_train = np.split(index, (20,), axis=0)
    #index_train = np.random.permutation(index_train)

    # Training
    for i in range(0, len(index_train), BATCH_SIZE):
        Text = []
        Label = []
        Feature = []
        for n in range(BATCH_SIZE):
            if i + n >= (len(index_train)):
                text = index_train[i + n - len(index_train)][0]
                label = index_train[i + n - len(index_train)][1]
                feature = index_train[i + n - len(index_train)][2]
            else:
                text = index_train[i + n][0]
                label = index_train[i + n][1]
                feature = index_train[i + n][2]
            Text.append(text)
            Label.append(label)
            Feature.append(feature)
        Text = np.array(Text, dtype="int32")
        Label = np.array(Label, dtype="int32")
        Feature = np.array(Feature, dtype="float32")

        Feature = np.mat(Feature)
        Feature = Feature.reshape(-1, 1)
        Feature = np.array(Feature)
        # print("feature vector = ", Feature)

        Feature = Variable(Feature)
        model.cleargrads()
        loss, predict_score = model(Text, Label, Feature)
        loss.backward()
        optimizer.update()

        losses_train.append(loss.data)
    # print(losses_train)
for i in range(0, len(index_test), BATCH_SIZE):
    Text = []
    Label = []
    Feature = []
    for n in range(BATCH_SIZE):
        if i + n >= (len(index_test)):
            text = index_test[i + n - len(index_test)][0]
            label = index_test[i + n - len(index_test)][1]
            feature = index_test[i + n - len(index_test)][2]
        else:
            text = index_test[i + n][0]
            label = index_test[i + n][1]
            feature = index_test[i + n][2]
        Text.append(text)
        Label.append(label)
        Feature.append(feature)
    Text = np.array(Text, dtype="int32")
    Label = np.array(Label, dtype="int32")
    Feature = np.array(Feature, dtype="float32")

    Feature = np.mat(Feature)
    Feature = Feature.reshape(-1, 1)
    Feature = np.array(Feature)
    # print("feature vector = ", Feature)

    Feature = Variable(Feature)
    model.cleargrads()
    loss, predict_score = model(Text, Label, Feature)
    #loss.backward()
    #optimizer.update()

    losses_test.append(loss.data)
    predict.append(predict_score.data)
print(losses_test)
save_data = pd.DataFrame(predict)
save_data.to_csv('predict_fold4.csv', sep='\t')
'''
    # Testing
    for i in range(0, len(index_test), BATCH_SIZE):
        Text = []
        Label = []
        Feature = []
        for n in range(BATCH_SIZE):
            if i + n >= (len(index_test)):
                text = index_test[i + n - len(index_test), 0]
                label = index_test[i + n - len(index_test), 1]
                feature = index_test[i + n - len(index_test), 2]
            else:
                text = index_test[i + n, 0]
                label = index_test[i + n, 1]
                feature = index_test[i + n, 2]
            Text.append(text)
            Label.append(label)
            Feature.append(feature)
        Text = np.array(Text, dtype="int32")
        Label = np.array(Label, dtype="int32")
        Feature = np.array(Feature, dtype="float32")

        Feature = np.mat(Feature)
        Feature = Feature.reshape(-1, 1)
        Feature = np.array(Feature)
        # print("feature vector = ", Feature)

        Feature = Variable(Feature)
        model.cleargrads()
        loss = model(Text, Label, Feature)
        #loss.backward()
        #optimizer.update()

        losses_test.append(loss.data)
    print(losses_test)
plt.plot(losses_train)
plt.plot(losses_test)
plt.show()
#chainer.serializers.save_npz("./xie.npz", model)
'''