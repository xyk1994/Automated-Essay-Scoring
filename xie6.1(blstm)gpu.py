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

xp = cuda.cupy
args = sys.argv


# モデルクラスの定義
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        super(LSTM_SentenceClassifier, self).__init__(
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.LSTM(embed_size, hidden_size),
            hy=L.Linear(hidden_size * 2, out_size)
        )

    # Forward propagation
    def __call__(self, text, label):
        # textを入力した際のネットワーク出力と、真値label との Rmse を返します。
        # print("text = ", text)
        # print("label = ", label)
        # print("feature = ", feature)
        x = F.transpose_sequence(text)
        label = xp.reshape(label, (len(label), 1))
        # feature = xp.reshape(feature, (len(feature), 1))
        self.eh.reset_state()

        # model---->
        for word in range(len(x)):
            # print("x[word] = ", (x[word]).shape)
            e = self.xe(x[word])
            # print("shape e = ", e.shape)
            h = self.eh(e)
            # print("shape h = ", h.shape)
        cel = h
        # cel = [10, 200]

        # <----model
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh(ee)
        cel_back = hh
        # cel_back = [10, 200]
        blstm = F.concat((cel, cel_back))       # blstm = [10, 200]
        # print("blstm = ", blstm)
        # print(type(blstm))
        # blstm_f = F.concat((blstm, feature))    # blstm_f = [10, 201]

        predict = self.hy(blstm)
        # predict = [10, 1]

        mse = F.mean_squared_error(predict, label)
        rmse = F.sqrt(mse)
        chainer.reporter.report({'loss': rmse}, self)

        return rmse

    def predictor2(self, text):
        x = F.transpose_sequence(text)
        # feature = xp.reshape(feature, (len(feature), 1))
        self.eh.reset_state()

        # model---->
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
        cel = h
        # cel = [10, 100]

        # <----model
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh(ee)
        cel_back = hh
        # cel_back = [10, 100]
        blstm = F.concat((cel, cel_back))       # blstm = [10, 200]
        # blstm_f = F.concat((blstm, feature))    # blstm_f = [10, 201]

        predict = self.hy(blstm)

        return predict


def sentence2words(sentence):
    data_tokensize = []
    stopWords = set(stopwords.words('english'))
    for i in range(len(sentence)):
        wordsFiltered = []
        words = word_tokenize(sentence[i])
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        data_tokensize.append(wordsFiltered)
    data_tokensize = np.array(data_tokensize)
    # print(data_tokensize)
    return data_tokensize


def make_datagroup(N, group_number):
    #データを分割して、データセットとテストセットに分けるための前処理
    # N データの個数
    # group_number 分割数(何グループに分けるか)
    print("make_datagroup")
    data_groups = []
    n = math.floor(N / group_number)
    for i in range(group_number - 1):
        data_group = list(range(n * i, n * (i + 1), 1))
        data_groups.append(data_group)
    data_group = list(range(n * (group_number - 1), N))
    data_groups.append(data_group)
    # print("fin make_datagroup")
    return data_groups


def create_dict(x):  # type(x) = np.array
    #辞書を作成する

    print("create_dict")
    vocab_dict = {}
    for sentence in x:
        for vocab in sentence:
            if vocab not in vocab_dict:
                vocab_dict[vocab] = len(vocab_dict)
    vocab_dict["UNK"] = len(vocab_dict)
    return vocab_dict


def get_max_sentence_size(data_x):
    #データセットの最大長を求める
    max_sentence_size = 0
    for sentence_vec in data_x:
        if max_sentence_size < len(sentence_vec):
            max_sentence_size = len(sentence_vec)
    print("max_sentence_size = ", max_sentence_size)
    return max_sentence_size


def to_word_idx_data(data_x, vocab_dict):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    print("word to index")
    data_x_vec = []
    for sentence in data_x:
        sentence_ids = []
        for word in sentence:
            sentence_ids.append(vocab_dict[word])
        data_x_vec.append(sentence_ids)
    return data_x_vec


def pedding_data(data_x_vec, max_sentence_size):
    for sentence_ids in data_x_vec:
        while len(sentence_ids) < max_sentence_size:
            sentence_ids.insert(0, -1)  # 先頭に追加
    return data_x_vec


def to_word_idx_test(test_x, vocab_dict):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    print("to_word_idx_test")
    test_x_vec = []
    for sentence in test_x:
        sentence_words = sentence2words(sentence)
        sentence_ids = []
        for word in sentence_words:
            if word not in vocab_dict:
                #　未知語対策
                sentence_ids.append(vocab_dict["UNK"])
            else:
                sentence_ids.append(vocab_dict[word])
        test_x_vec.append(sentence_ids)
    return test_x_vec


def pedding_test(test_x_vec, max_sentence_size):
    for sentence_ids in test_x_vec:
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
    return test_x_vec


csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature(300).tsv", delimiter="\t", engine="python", error_bad_lines=False)
csv_input = csv_input[["essay", "rater1_domain1"]]
data = csv_input

# SEED=int(args[3]) seedの値を入力で決める場合
SEED = 1  # seed=1に固定した
xp.random.seed(SEED)
random.seed(SEED)

N = len(data)
data_text = []
for d in range(N):
    data_text.append(data.ix[d, 0])

words = sentence2words(data_text)
vocab_dict = create_dict(words)


data_groups = make_datagroup(N, 5)
Score = []
Text = []

# 推定の答えを格納するリスト
for group_number in range(5):
    print("group number = ", group_number)
    data_x, data_t = [], []
    test_x, test_t = [], []  # テストデータを別にする

    for d in range(N):
        # print(d)
        if d in data_groups[group_number]:
            test_x.append(words[d])  # 文書
            test_t.append(data.ix[d, 1])  # ラベル
            # test_f.append(data.ix[d, 2])  # feature
        else:
            data_x.append(words[d])  # 文書
            data_t.append(data.ix[d, 1])  # ラベル
            # data_f.append(data.ix[d, 2])  # feature
    print("fin data_x, data_t")

    data_x_vec = to_word_idx_data(data_x, vocab_dict)
    test_x_vec = to_word_idx_data(test_x, vocab_dict)
    max_sentence_size = get_max_sentence_size(data_x_vec)  # データセットの最大長を求める
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    data_x_vec = pedding_data(data_x_vec, max_sentence_size)
    test_x_vec = pedding_test(test_x_vec, max_sentence_size)

    print("len(data_x_vec) = ", len(data_x_vec))
    # print("data_x_vec.size = ", data_x_vec.size)
    print("len(test_x_vec) = ", len(test_x_vec))
    # print("test_x_vec.size = ", test_x_vec.size)
    # データセット
    data_x_vec = xp.array(data_x_vec, dtype=xp.int32)
    test_x_vec = xp.array(test_x_vec, dtype=xp.int32)
    data_t = xp.array(data_t, dtype=xp.float32)
    test_t = xp.array(test_t, dtype=xp.float32)
    # data_f = xp.array(data_f, dtype=xp.float32)
    # test_f = xp.array(test_f, dtype=xp.float32)
    # dataset = []
    # for x, t in zip(data_x_vec, data_t):
    #    dataset.append((x, t))
    dataset = list(zip(data_x_vec, data_t))

    for sentence in test_x_vec:
        for id in sentence:
            if id < -1:
                print("Error")
            elif id >= len(vocab_dict):
                print("Error2")

    # モデルの定義
    print("define model")
    # model = MyRegressor(LSTM_SentenceClassifier(
    model = LSTM_SentenceClassifier(
        vocab_size=len(vocab_dict),
        embed_size=100,
        hidden_size=100,
        out_size=1,
    )
    # cuda.get_device(0).use()
    model.to_gpu(0)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    print("optimizer setted")
    # 学習開始
    EPOCH_NUM = 50
    BATCH_SIZE = 10
    # chainer.config.cudnn_deterministic = True
    train, dev = chainer.datasets.split_dataset_random(dataset, len(dataset) - 20, seed=SEED)
    train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE, shuffle=False)
    dev_iter = chainer.iterators.SerialIterator(dev, BATCH_SIZE, repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch',
                                         file_name='AES3 rater1_domain1 accuracy' + str(group_number) + '.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(dev_iter, model, device=0))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", "validation/main/loss", "elapsed_time"], ))  # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
    trainer.extend(extensions.ProgressBar())  # プログレスバー出力
    trainer.run()

    print("training finished")
    print("test start")
    predict = model.predictor2(test_x_vec)
    score = predict.data
    score = [s[0] for s in score]
    print(score)
    Score.append(score)
    Text.append(test_x)

Score = pd.DataFrame(Score)
Score.to_csv('data/Score(blstm).csv', sep=',')

Text = pd.DataFrame(Text)
Text.to_csv('data/Text(blstm).csv', sep=',')