#from janome.tokenizer import Tokenizer
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

xp = cuda.cupy
# プログラム実行時に69行目data = csv_input[["essay","rater1_domain1"]]の"eaasy"と"rater1_domain1を決める"
args = sys.argv


# モデルクラスの定義
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        super(LSTM_SentenceClassifier, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = L.LSTM(embed_size, hidden_size),
            hy = L.Linear(hidden_size * 2, out_size)

        )

    # Forward propagation
    def __call__(self, x,t):
        # xを入力した際のネットワーク出力と、回答t との差分を返します。
        x = F.transpose_sequence(x)
        self.eh.reset_state()

        # model---->
        #return self.predict(h)
        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
        cel = h

        # <----model
        for word in range(1, len(x)):
            ee = self.xe(x[len(x) - word])
            hh = self.eh(ee)
        cel_back = hh

        blstm = F.concat((cel, cel_back))

        predict = self.hy(blstm)


        #print(y)
        label = xp.reshape(t, (len(t), 1))
        #print(t)
        mse = F.mean_squared_error(predict, label)
        rmse = F.sqrt(mse)
        chainer.reporter.report({'loss': rmse}, self)

        return rmse

    def predictor2(self,x):
        x = F.transpose_sequence(x)
        self.eh.reset_state()
        #return self.predict(h)

        for word in range(len(x)):
            e = self.xe(x[word])
            h = self.eh(e)
            #print(h)
        #y = F.leaky_relu(self.hy(h))
        y = self.hy(h)
        return y
        #y = self.hy(h)
        #return F.softmax(y)


def forward(x,  model):
    t = model.predict(x[0][0])
    loss = F.mean_squared_error(t, x[0][1])
    return loss


def sentence2words(sentence):
    stopwords = ["i", "a", "an", "the", "and", "or", "if", "is", "are", "am", "it", "this", "that", "of", "from", "in", "on"],
    sentence = sentence.lower() # 小文字化
    sentence = sentence.replace("\n", "") # 改行削除
    sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~],"), " ", sentence) # 記号をスペースに置き換え
    sentence = sentence.split(" ") # スペースで区切る
    sentence_words = []
    for word in sentence:
        if word in stopwords: # ストップワードに含まれるものは除外
            continue
        sentence_words.append(word)
    return sentence_words

#t = Tokenizer()
def sentence2words_japanese(sentence):
    t = Tokenizer()
    #sentence = sentence.replace("\n", "") # 改行削除
    #sentence = sentence.split(" ") # スペースで区切る
    sentence_words = []
    sentence = t.tokenize(sentence,wakati=True)
    for word in sentence:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        sentence_words.append(word)
    return sentence_words

def make_datagroup(N,group_number):
    #データを分割して、データセットとテストセットに分けるための前処理
    # N データの個数
    # group_number 分割数(何グループに分けるか)
    print("start make_datagroup")
    data_groups=[]
    n = math.floor(N/group_number)
    for i in range(group_number-1):
        data_group=list(range(n*i,n*(i+1),1))
        data_groups.append(data_group)
    data_group=list(range(n*(group_number-1),N))
    data_groups.append(data_group)
    print("fin make_datagroup")
    return data_groups

def create_dict(x):
    #辞書を作成する
    print("start create_dict")
    vocab_dict = {}
    for sentence in x:
        for vacab in sentence2words(sentence):
            if vacab not in vocab_dict:
                vocab_dict[vacab] = len(vocab_dict)
    vocab_dict["UNK"] = len(vocab_dict)
    print("fin create_dict")
    return vocab_dict

def get_max_sentence_size(data_x):
    #データセットの最大長を求める
    max_sentence_size = 0
    for sentence_vec in data_x:
        if max_sentence_size < len(sentence_vec):
            max_sentence_size = len(sentence_vec)
    return max_sentence_size

def to_word_idx_data(data_x, vocab_dict, max_sentence_size):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    print("start to_word_idx_data")
    data_x_vec = []
    for sentence in data_x:
        sentence_words = sentence2words(sentence)
        #sentence_words = sentence2words_japanese(sentence)
        sentence_ids = []
        for word in sentence_words:
            sentence_ids.append(vocab_dict[word])
        data_x_vec.append(sentence_ids)

    for sentence_ids in data_x_vec:
        while len(sentence_ids) < max_sentence_size:
            sentence_ids.insert(0, -1)
    print("fin to_word_idx_data")
    return xp.array(data_x_vec, dtype=xp.int32)

def to_word_idx_test(test_x, vocab_dict, max_sentence_size):
    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    print("start to_word_idx_test")
    test_x_vec = []
    for sentence in test_x:
        sentence_words = sentence2words(sentence)
        #sentence_words = sentence2words_japanese(sentence)
        sentence_ids = []
        for word in sentence_words:
            if word not in vocab_dict:
                #　未知語対策
                sentence_ids.append(vocab_dict["UNK"])
            else:
                sentence_ids.append(vocab_dict[word])
        test_x_vec.append(sentence_ids)

    for sentence_ids in test_x_vec:
        if len(sentence_ids) < max_sentence_size:
            while len(sentence_ids) < max_sentence_size:
                sentence_ids.insert(0, -1) # 先頭に追加
        if len(sentence_ids) > max_sentence_size:
            while len(sentence_ids) > max_sentence_size:
                if vocab_dict["UNK"] in sentence_ids:
                    # UNKの箇所を先頭の方から削除
                    sentence_ids.pop(sentence_ids.index(vocab_dict["UNK"]))
                else:
                    sentence_ids.pop(0) # 先頭を消去
    print("fin to_word_idx_test")
    return xp.array(test_x_vec, dtype=xp.int32)

#SEED=int(args[3]) seedの値を入力で決める場合
SEED=1 #seed=1に固定した
xp.random.seed(SEED)
random.seed(SEED)

# csv_input = pd.read_csv(filepath_or_buffer="../data/Japanese_IRT.csv", encoding="UTF-8", sep=",")
csv_input = pd.read_csv(filepath_or_buffer="essayset1_with_feature (300).tsv", delimiter="\t", engine="python", error_bad_lines=False)
data = csv_input[["essay", "rater1_domain1"]]
# data = csv_input[[args[1],args[2]]]
#print(data)
#OUT_SIZE = max(data.loc[:,args[2]].values) + 1
OUT_SIZE = 1
#print(uni)
N = len(data)
print(N)
#データを10個のグループに分けておく
data_groups=make_datagroup(N,10)
print(data_groups)
fin_ans=[]
#推定の答えを格納するリスト
for group_number in range(10):
    print(group_number)
    print(data_groups[group_number])
    data_x, data_t = [], []
    test_x, test_t = [], [] #テストデータを別にする

    for d in range(N):
        #print(d)
        if d in data_groups[group_number]:
            #print(d)
            continue
        data_x.append(data.iloc[d, 0]) # 文書
        data_t.append(data.iloc[d, 1]) # ラベル
    print("fin data_x,data_t")
    for d in data_groups[group_number]:
        test_x.append(data.iloc[d,0]) # 文書
        test_t.append(data.iloc[d,1]) # ラベル
    vocab_dict = create_dict(data_x) #単語辞書を作成する
    max_sentence_size = get_max_sentence_size(data_x) #データセットの最大長を求める

    # 文章を単語ID配列にする & 文章の長さを揃えるために-1を前パディング
    data_x_vec = to_word_idx_data(data_x, vocab_dict, max_sentence_size)
    test_x_vec = to_word_idx_test(test_x, vocab_dict, max_sentence_size)
    print(len(data_x_vec))
    print(data_x_vec.size)
    print(len(test_x_vec))
    print(test_x_vec.size)
    # データセット
    data_x_vec = xp.array(data_x_vec, dtype=xp.int32)
    test_x_vec = xp.array(test_x_vec, dtype=xp.int32)
    data_t = xp.array(data_t, dtype=xp.float32)
    test_t = xp.array(test_t, dtype=xp.float32)
    dataset = []
    for x, t in zip(data_x_vec, data_t):
        dataset.append((x, t))

    for sentence in test_x_vec:
        for id in sentence:
            if id < -1 :
                print("Error")
            elif id >= len(vocab_dict):
                print("Error2")

    # モデルの定義
    print("def model")
#    model = MyRegressor(LSTM_SentenceClassifier(
    model = LSTM_SentenceClassifier(
        vocab_size=len(vocab_dict),
        embed_size=100,
        hidden_size=100,
        out_size=OUT_SIZE,

    #), lossfun=mean_squared_error)
    )
    cuda.get_device(0).use()
    model.to_gpu(0)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    print("optimizer set")
    # 学習開始
    EPOCH_NUM = 50
    BATCH_SIZE = 10
#    chainer.config.cudnn_deterministic = True
    train, dev = chainer.datasets.split_dataset_random(dataset, len(dataset)-20,seed=SEED)
    train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE, shuffle=False)
    dev_iter = chainer.iterators.SerialIterator(dev, BATCH_SIZE, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name= 'AES3 rater1_domain1 accuracy'+str(group_number)+'.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(dev_iter, model, device=0))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"],)) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
    trainer.extend(extensions.ProgressBar()) # プログレスバー出力
    trainer.run()



#    print(train_iter.next())
#    for i in range(0,1000):
#        print(i)
#        a=train_iter.next()
#        print(a[0][0])
#        print(a[0][1])
#        loss = forward(train_iter.next(), model)
#        print(loss.data)  # 現状のMSEを表示
#        optimizer.update(forward, x, y, model)



    #chainer.serializers.save_npz("mymodel.npz", model)

    #test
    print("test start")

    xt = Variable(test_x_vec)
#    print(xt)
    yt = model.predictor2(xt)
    ans = yt.data
    print(ans)
    fin_ans.append(ans)
    print(test_t)
print(fin_ans)
