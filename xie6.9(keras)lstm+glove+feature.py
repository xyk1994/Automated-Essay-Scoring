from keras.models import Model, Sequential, load_model, model_from_config
from keras.layers import Input, concatenate,InputSpec, Activation, Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Bidirectional
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


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


def get_clean_essays(Text):
    clean_essays = []
    for essay in Text:
        clean_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
    return clean_essays


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
    text_input = Input(shape=(max_essay_length, ), name='text_input')
    feature_input = Input(shape=(num_feature, ), name='feature_input')
    # model = Sequential()
    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=100,
                                weights=[embeddings_matrix],
                                input_length=max_essay_length,
                                trainable=False,
                                )
    embedding = embedding_layer(text_input)
    lstm_out = LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False)(embedding)
    concat_layer = concatenate([lstm_out, feature_input])
    dense1 = Dense(64, activation='tanh')(concat_layer)
    dense2 = Dense(64, activation='tanh')(dense1)
    output = Dense(1, activation='relu')(dense2)
    model = Model(inputs=[text_input, feature_input], outputs=output)
    model.summary()
    return model


csv_input = pd.read_csv(filepath_or_buffer="essay_set7_with_feature.csv", delimiter=",", engine="python", error_bad_lines=False)
Score = csv_input['domain1_score']
Text = csv_input['essay']
Feature = csv_input[['word_count', 'sent_count', 'avg_word_len', 'lemma_count', 'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 'adv_count']]
num_feature = 9
# csv_input = csv_input.dropna(axis=1)
# data = csv_input.drop(columns=['rater1_domain1', 'rater2_domain1'])
# Text = data['essay']

essays = get_clean_essays(Text)

max_essay_length = max_essay_length(essays)

word_index = create_dict(essays)

essays_in_index = np.array(essay_to_index(essays, word_index))
essays_in_index = pad_sequences(essays_in_index, maxlen=max_essay_length)

embeddings_index = get_embeddings_index()
embeddings_matrix = embeddings_matrix(embeddings_index, word_index, embedding_dim=100)

results = []
score_predict_list = []
count = 1
cv = KFold(n_splits=5, shuffle=True)
for train, test in cv.split(essays_in_index):
    text_train, text_test = essays_in_index[train], essays_in_index[test]
    label_train, label_test = Score.iloc[train], Score.iloc[test]
    feature_train, feature_test = Feature.iloc[train], Feature.iloc[test]


    model = get_model()
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.fit([text_train, feature_train], label_train, batch_size=64, epochs=50)

    score_predict = model.predict([text_test, feature_test])
    score_predict = np.around(score_predict)
    result = cohen_kappa_score(label_test.values, score_predict, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))

