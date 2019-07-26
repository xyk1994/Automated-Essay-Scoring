import re
import sys
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
import math
import fileinput
from matplotlib import pyplot as plt
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions

xp = cuda.cupy
args = sys.argv


# モデルクラスの定義
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        super(LSTM_SentenceClassifier, self).__init__(
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.LSTM(embed_size, hidden_size),
            hy=L.Linear(hidden_size, out_size)

        )

    # Forward propagation
    def __call__(self, text, label, feature):
        # textを入力した際のネットワーク出力と、真値label との Rmse を返します。
        x = F.transpose_sequence(text)
        self.eh.reset_state()

        # model---->
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
        cel = h
        # cel = [10, 200]

        # <----model
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh(ee)
        cel_back = hh
        # cel_back = [10, 200]
        blstm = F.concat((cel, cel_back))       # blstm = [10, 400]
        blstm_f = F.concat((blstm, feature))    # blstm_f = [10, 401]

        predict = self.hy(blstm_f)
        # predict = [10, 1]

        label = xp.reshape(label, (len(label), 1))

        mse = F.mean_squared_error(predict, label)
        rmse = F.sqrt(mse)
        chainer.reporter.report({'loss': rmse}, self)

        return rmse

    def predictor2(self, text):
        x = F.transpose_sequence(text)
        self.eh.reset_state()

        # model---->
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
        cel = h
        # cel = [10, 200]

        # <----model
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh(ee)
        cel_back = hh
        # cel_back = [10, 200]
        blstm = F.concat((cel, cel_back))       # blstm = [10, 400]
        blstm_f = F.concat((blstm, feature))    # blstm_f = [10, 401]

        predict = self.hy(blstm_f)

        return predict


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


def get_dataset(dataset, n_fold, fold):
    index = chainer.datasets.sub_dataset.get_cross_validation_datasets(dataset, n_fold, order=None)
    index_loop = index[fold]
    index_train = index_loop[0]
    index_test = index_loop[1]
    return index_train, index_test


def create_dict(dataset):
    # 辞書を作成する
    # print("Create dictionary")
    vocab_dict = {}
    for sentence in dataset:
        for word in sentence2words(sentence):
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)
    vocab_dict["UNK"] = len(vocab_dict)
    print("Successful created dictionary")
    return vocab_dict


def get_max_sentence_size(dataset):
    # データセットの最大長を求める
    max_sentence_size = 0
    for sentence_vec in dataset:
        if max_sentence_size < len(sentence_vec):
            max_sentence_size = len(sentence_vec)
    print("max_sentence_size = ", max_sentence_size)
    return max_sentence_size


def to_word_idx_data(data_train, vocab_dict, max_sentence_size):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    data_train_vec = []
    for sentence in data_train:
        sentence_words = sentence2words(sentence)
        sentence_ids = []
        for word in sentence_words:
            sentence_ids.append(vocab_dict[word])
        data_train_vec.append(sentence_ids)

    for sentence_ids in data_train_vec:
        while len(sentence_ids) < max_sentence_size:
            sentence_ids.insert(0, -1)
    print("Successful to word idx data_train")
    return xp.array(data_train_vec, dtype=xp.int32)


def to_word_idx_test(data_test, vocab_dict, max_sentence_size):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    data_test_vec = []
    for sentence in data_test:
        sentence_words = sentence2words(sentence)
        sentence_ids = []
        for word in sentence_words:
            if word not in vocab_dict:
                #  未知語対策
                sentence_ids.append(vocab_dict["UNK"])
            else:
                sentence_ids.append(vocab_dict[word])
        data_test_vec.append(sentence_ids)

    for sentence_ids in data_test_vec:
        if len(sentence_ids) < max_sentence_size:
            while len(sentence_ids) < max_sentence_size:
                sentence_ids.insert(0, -1)  # 先頭に追加
        if len(sentence_ids) > max_sentence_size:
            while len(sentence_ids) > max_sentence_size:
                if vocab_dict["UNK"] in sentence_ids:
                    # UNKの箇所を先頭の方から削除
                    sentence_ids.pop(sentence_ids.index(vocab_dict["UNK"]))
                else:
                    sentence_ids.pop(0)  # 先頭を消去
    print("Successful to word idx data_test")
    return xp.array(data_test_vec, dtype=xp.int32)


print("----------------------")
print("|AES_with_word_length|")
print("----------------------")

csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature(300).tsv", delimiter="\t", engine="python", error_bad_lines=False)
csv_input = csv_input[["essay", "rater1_domain1", "word_length"]]
data = csv_input

data_text, data_label, data_feature = [], [], []
for d in range(len(data)):
    data_text.append(data.ix[d, 0])
    data_label.append(data.ix[d, 1])
    data_feature.append(data.ix[d, 2])
vocab_dict = create_dict(data_text)     # 単語辞書を作成する
max_sentence_size = get_max_sentence_size(data_text)
data_text_vec = to_word_idx_data(data_text, vocab_dict, max_sentence_size)

#data_text_vec = np.array(data_text_vec, dtype="int32")  # text
data_label = np.array(data_label, dtype="int32")  # label
data_feature = np.array(data_feature, dtype="int32")  # feature
dataset = list(zip(data_text_vec, data_label, data_feature))

# 定数
EPOCH_NUM = 50
EMBED_SIZE = 200
HIDDEN_SIZE = 200
BATCH_SIZE = 10
n_fold = 5
fold = 0
OUT_SIZE = 1

data_train, data_test = get_dataset(dataset, n_fold=n_fold, fold=fold)

'''
train_text, train_label, train_feature = [], [], []
test_text, test_label, test_feature = [], [], []


for d in range(len(data_train)):
    train_text.append(data_train[d, 0])      # 文書
    train_label.append(data_train[d, 1])     # ラベル
    train_feature.append(data_train[d, 2])   # 特徴量
for d in range(len(data_test)):
    test_text.append(data_test[d, 0])        # 文書
    test_label.append(data_test[d, 1])       # ラベル
    test_feature.append(data_test[d, 2])     # 特徴量


train_text_vec = to_word_idx_data(train_text, vocab_dict, max_sentence_size)
test_text_vec = to_word_idx_test(test_text, vocab_dict, max_sentence_size)

train_text_vec = xp.array(train_text_vec, dtype=xp.int32)
test_text_vec = xp.array(test_text_vec, dtype=xp.int32)
train_label = xp.array(train_label, dtype=xp.float32)
test_label = xp.array(test_label, dtype=xp.float32)
train_feature = xp.array(train_feature, dtype=xp.int32)
test_feature = xp.array(test_feature, dtype=xp.int32)


for sentence in test_text_vec:
    for index in sentence:
        if id < -1 or id >= len(vocab_dict):
            print("Value Error: to_word_idx_test()")
'''

model = LSTM_SentenceClassifier(vocab_size=len(vocab_dict),
                                embed_size=EMBED_SIZE,
                                hidden_size=HIDDEN_SIZE,
                                out_size=OUT_SIZE,
                                )

# cuda.get_device(0).use()
model.to_gpu(0)
optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = iterators.SerialIterator(data_train, BATCH_SIZE, shuffle=False)
test_iter = iterators.SerialIterator(data_test, BATCH_SIZE, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name= 'AES3'+args[2]+'accuracy'+str(group_number)+'.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"],))  # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.ProgressBar())  # プログレスバー出力
trainer.run()




