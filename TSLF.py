from keras.models import Model, Sequential, load_model, model_from_config
from keras.layers import Input, concatenate,InputSpec, Activation, Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Bidirectional
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import pandas as pd
import numpy as np
import nltk, re, collections
from collections import defaultdict
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


'''
# Feature and GBDT
# Tokenize a sentence into words
def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)

    return tokens


# tokenizing an essay into a list of word lists
def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    return tokenized_sentences


# calculating average word length in an essay
def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return sum(len(word) for word in words) / len(words)


# calculating number of words in an essay
def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return len(words)


# calculating number of characters in an essay
def char_count(essay):
    clean_essay = re.sub(r'\s', '', str(essay).lower())

    return len(clean_essay)


# calculating number of sentences in an essay
def sent_count(essay):
    sentences = nltk.sent_tokenize(essay)

    return len(sentences)


# calculating number of lemmas per essay
def count_lemmas(essay):
    tokenized_sentences = tokenize(essay)

    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count


# checking number of misspelled words
def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    # big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.
    data = open('dataset/big.txt').read()

    words_ = re.findall('[a-z]+', data.lower())

    word_dict = collections.defaultdict(lambda: 0)

    for word in words_:
        word_dict[word] += 1

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in word_dict:
            mispell_count += 1

    return mispell_count


# calculating number of nouns, adjectives, verbs and adverbs in an essay
def count_pos(essay):
    tokenized_sentences = tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1

    return noun_count, adj_count, verb_count, adv_count


# getiing Bag of Words (BOW) counts
def get_count_vectors(essays):
    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')

    count_vectors = vectorizer.fit_transform(essays)

    feature_names = vectorizer.get_feature_names()

    return feature_names, count_vectors


# extracting essay features
def extract_features(data):
    features = data.copy()

    features['char_count'] = features['essay'].apply(char_count)

    features['word_count'] = features['essay'].apply(word_count)

    features['sent_count'] = features['essay'].apply(sent_count)

    features['avg_word_len'] = features['essay'].apply(avg_word_len)

    features['lemma_count'] = features['essay'].apply(count_lemmas)

    features['spell_err_count'] = features['essay'].apply(count_spell_error)

    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(
        *features['essay'].map(count_pos))

    return features

'''


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
    # feature_input = Input(shape=(num_feature, ), name='feature_input')
    # model = Sequential()
    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=100,
                                weights=[embeddings_matrix],
                                input_length=max_essay_length,
                                trainable=False
                                )
    embedding = embedding_layer(text_input)
    lstm_out = LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False)(embedding)
    # lstm_out = Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))(embedding)
    # concat_layer = concatenate([lstm_out, feature_input])
    #dense1 = Dense(64, activation='tanh')(lstm_out)
    #dense2 = Dense(32, activation='tanh')(dense1)
    output = Dense(1, activation='sigmoid')(lstm_out)
    model = Model(inputs=text_input, outputs=output)
    model.summary()
    return model


csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)
# csv_input = pd.read_csv(filepath_or_buffer="essay_set7_with_feature.csv", delimiter=",", engine="python", error_bad_lines=False)
Score = csv_input['domain1_score']
Hand_marked_score = csv_input['hand_marked_score']
Text = csv_input['essay']

essays = get_clean_essays(Text)

max_essay_length = max_essay_length(essays)

word_index = create_dict(essays)

essays_in_index = np.array(essay_to_index(essays, word_index))
essays_in_index = pad_sequences(essays_in_index, maxlen=max_essay_length)

embeddings_index = get_embeddings_index()
embeddings_matrix = embeddings_matrix(embeddings_index, word_index, embedding_dim=100)

results = []
score_predict_list = []
text_word = []
true_score = []
count = 1
cv = KFold(n_splits=5, shuffle=True)
for train, test in cv.split(essays_in_index):
    text_train, text_test = essays_in_index[train], essays_in_index[test]
    label_train, label_test = Score.iloc[train], Score.iloc[test]
    hand_marked_train, hand_marked_test = Hand_marked_score.iloc[train], Hand_marked_score[test]
    word_train, test_word = Text.iloc[train], Text.iloc[test]

    model = get_model()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(text_train, hand_marked_train, batch_size=64, epochs=100)

    score_predict = model.predict(text_test)
    Score_predict = (10 * score_predict) + 2
    Score_predict = np.around(Score_predict)  # 用来算qwk的score
    score_predict = [s[0] for s in score_predict]
    for score in score_predict:
        score_predict_list.append(score)
    #print(score_predict)
    test_word = np.array(test_word)
    for essay in test_word:
        text_word.append(essay)
    Label_test = np.array(label_test)
    for score in Label_test:
        true_score.append(score)
    #print('score_predict = ', score_predict)
    #print(score_predict.shape)

    print('-----------------------------------------')
    result = cohen_kappa_score(label_test.values, Score_predict, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))
'''
predict_score = []
for fold in score_predict_list:
    for score in fold:
        predict_score.append(score)
predict_score = np.array(predict_score)
essay_list = []
for fold in text_word:
    for essay in text_word:
        essay_list.append(essay)
essay_list = np.array(essay_list)
'''
data = zip(text_word, true_score, score_predict_list)
data = list(data)
data = pd.DataFrame(data)
data.to_csv('data/predict_data_set1(6.14).tsv', sep='\t')
