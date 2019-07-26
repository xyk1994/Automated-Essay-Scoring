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


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)

DATASET_DIR = './dataset/'
X = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
# X = pd.read_csv(filepath_or_buffer="essay_set1_with_feature(300).tsv", delimiter="\t", engine="python", error_bad_lines=False)
Label = X['domain1_score']
X = X.dropna(axis=1)
X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])
Text = X['essay']

print(X.head())

results = []
Score_pred_list = []

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
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


def get_model():
    """Define the model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(100, dropout=0.4, recurrent_dropout=0.4, return_sequences=True), input_shape=[1, 100]))
    # model.add(LSTM(64, recurrent_dropout=0.4))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

print("Loading Glove Model")
model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)

'''
sentences = []
for essay in Text:
    sentences += essay_to_sentences(essay, remove_stopwords=True)
'''
print("Preprocessing the Data")
num_features = 100  # word embeddings size
clean_essays = []
for essay in Text:
    clean_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
DataVecs = getAvgFeatureVecs(clean_essays, model, num_features)
DataVecs = np.array(DataVecs)

count = 1

cv = KFold(n_splits=5, shuffle=False)   # Cross Validation n_folds = 5
for traincv, testcv in cv.split(DataVecs):
    print("\n--------Fold {}--------\n".format(count))
    DataVecs_train, DataVecs_test, Label_train, Label_test = \
        DataVecs[traincv], DataVecs[testcv], Label.iloc[traincv], Label.iloc[testcv]
    DataVecs_train = np.array(DataVecs_train)
    DataVecs_test = np.array(DataVecs_test)
    DataVecs_train = np.reshape(DataVecs_train, (DataVecs_train.shape[0], 1, DataVecs_train.shape[1]))
    DataVecs_test = np.reshape(DataVecs_test, (DataVecs_test.shape[0], 1, DataVecs_test.shape[1]))

    print("Get Model")
    lstm_model = get_model()
    lstm_model.fit(DataVecs_train, Label_train, batch_size=64, epochs=50)
    score_pred = lstm_model.predict(DataVecs_test)

    if count == 5:
        print("Save Model")
        lstm_model.save('./model_weights/final_lstm.h5')

    Score_pred_list = np.around(score_pred)

    result = cohen_kappa_score(Label_test.values, Score_pred_list, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))
