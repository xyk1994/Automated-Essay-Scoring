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
# X = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
X = pd.read_csv(filepath_or_buffer="essay_set_1.tsv", delimiter="\t", engine="python", error_bad_lines=False)
y = X['domain1_score']
X = X.dropna(axis=1)
X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])
print(X.head())



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


def get_model():
    """Define the model."""
    model = Sequential()
    # model.add(Bidirectional(LSTM(100, dropout=0.4, recurrent_dropout=0.4, return_sequences=True), input_shape=[1, 100]))
    #model.add(Bidirectional(LSTM(100, dropout=0.4, recurrent_dropout=0.4, return_sequences=True), merge_mode='concat', input_shape=[1, 100]))
    model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1, return_sequences=False, input_shape=[1, 100]))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model
'''

def get_model():
    """Define the model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=[1, 300]))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

'''
kf = KFold(n_splits=5, shuffle=True)
results = []
y_pred_list = []

count = 1

# model = KeyedVectors.load_word2vec_format('glove.6B.300d.word2vec.txt', binary=False)
# print("Loading Glove Model")

for traincv, testcv in kf.split(X):
    print("\n--------Fold {}--------\n".format(count))
    X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    train_essays = X_train['essay']
    test_essays = X_test['essay']

    sentences = []
    print("Train essay to sentences")
    for essay in train_essays:
        # Obtaining all sentences from the training essays.
        sentences += essay_to_sentences(essay, remove_stopwords=True)

    # Initializing variables for word2vec model.
    num_features = 100
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3
    
    print("Training Word2Vec Model...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling)

    model.init_sims(replace=True)
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)



    # Generate training and testing data word vectors.
    print("Generate training and testing data word vectors.")
    clean_train_essays = []
    for essay_v in train_essays:
        clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)

    clean_test_essays = []
    for essay_v in test_essays:
        clean_test_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(clean_test_essays, model, num_features)

    trainDataVecs = np.array(trainDataVecs)
    testDataVecs = np.array(testDataVecs)
    # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
    print(trainDataVecs.shape)
    print(testDataVecs.shape)
    print("Get Model")
    lstm_model = get_model()
    lstm_model.fit(trainDataVecs, y_train, batch_size=32, epochs=200)
    # lstm_model.load_weights('./model_weights/final_lstm.h5')
    y_pred = lstm_model.predict(testDataVecs)

    # Save any one of the 8 models.
    if count == 5:
        print("Save Model")
        lstm_model.save('./model_weights/final_lstm.h5')

    # Round y_pred to the nearest integer.
    y_pred_list = np.around(y_pred)

    # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
    result = cohen_kappa_score(y_test.values, y_pred_list, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))

Score = pd.DataFrame(y_pred_list)
Score.to_csv('data/Score(keras).csv', sep=',')