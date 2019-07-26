from keras.models import Model
from keras.layers import LSTM, Dense, Input, Embedding, Concatenate,InputSpec, Activation, Lambda
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Bidirectional
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences


def get_model(max_len, input_dim, lstm_op_dim):
    inputs = Input(shape=(max_len, input_dim))
    lstm = LSTM(lstm_op_dim, return_sequences=False)(inputs)
    output = Dense(1, activation='relu')(lstm)
    model = Model(inputs=inputs, outputs=output)
    return model


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


def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec


def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


def max_essay_length(data):
    max_essay_length = 0
    for essay in data:
        if max_essay_length < len(essay):
            max_essay_length = len(essay)
    return max_essay_length


def essay_to_wordvec(data):
    DataVecs = []
    for essay in data:
        EssayVecs = []
        for word in essay:
            EssayVecs.append(model[word])
        DataVecs.append(EssayVecs)
    return DataVecs


def create_dict(text):
    print("create_dict")
    word_index = {}
    for sentence in text:
        for word in sentence:
            if word not in word_index:
                word_index[word] = len(word_index)
    word_index['unk'] = len(word_index)
    return word_index


def essay_to_index(text, word_index):
    text_index = []
    for essay in text:
        essay_index = []
        for word in essay:
            essay_index.append(word_index[word])
        text_index.append(essay_index)
    return text_index


def get_embeddings_index():
    embeddings_index = {}
    f = open('glove.6B.100d.txt', 'r', encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def embeddings_matrix(embeddings_index, word_index, embedding_dim):
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embedding_matrix[i] = embeddings_vector
    return embedding_matrix


def get_model():
    print('Build model...')
    model = Sequential()
    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=100,
                                weights=[embeddings_matrix],
                                input_length=max_essay_length,
                                trainable=False,
                                # dropout=0.2
                                )
    model.add(embedding_layer)
    model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model


csv_input = pd.read_csv(filepath_or_buffer="essay_set_1.tsv", delimiter="\t", engine="python", error_bad_lines=False)
Score = csv_input['domain1_score']
csv_input = csv_input.dropna(axis=1)
data = csv_input.drop(columns=['rater1_domain1', 'rater2_domain1'])
Text = data['essay']

# print(Text)

sentences = []
print("Train essay to sentences")
for essay in Text:
    # Obtaining all sentences from the training essays.
    sentences += essay_to_sentences(essay, remove_stopwords=True)

num_features = 50
min_word_count = 1
num_workers = 4
context = 10
downsampling = 1e-3

# print(np.array(sentences).shape)
'''
print("Training Word2Vec Model...")
model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                 sample=downsampling)

model.init_sims(replace=True)
model.wv.save_word2vec_format('word2vecmodel.txt', binary=False)
'''

clean_essays = []
for essay_v in Text:
    clean_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
# print(clean_essays)
# clean_essays：论文的单词list[论文数,单词数]

max_essay_length = max_essay_length(clean_essays)
# print(max_essay_length)

word_index = create_dict(clean_essays)
# print("word_index['computer'] = ", word_index['computer'])

text_in_index = essay_to_index(clean_essays, word_index)
# print(np.array(text_with_index)) [文章数，单词数]

text_in_index = np.array(text_in_index)
text_in_index = pad_sequences(text_in_index, maxlen=max_essay_length)
# print(text_with_index)
# print(text_with_index.shape)

embeddings_index = get_embeddings_index()
embeddings_matrix = embeddings_matrix(embeddings_index, word_index, embedding_dim=100)
# print(embeddings_matrix.shape)

results = []
score_predict_list = []
count = 1
cv = KFold(n_splits=5, shuffle=True)
for train, test in cv.split(text_in_index):
    text_train, text_test = text_in_index[train], text_in_index[test]
    label_train, label_test = Score.iloc[train], Score.iloc[test]

    lstm_model = get_model()
    lstm_model.fit(text_train, label_train, batch_size=64, epochs=10)
    score_predict = lstm_model.predict(text_test)

    if count == 5:
        print("Save Model")
        lstm_model.save('./model_weights/final_lstm.h5')

    score_predict = np.around(score_predict)
    result = cohen_kappa_score(label_test.values, score_predict, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))