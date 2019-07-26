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
from nltk.tokenize import word_tokenize, sent_tokenize
import keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from gensim.models import Word2Vec



def get_clean_essays(Text):
    clean_essays = []
    for essay in Text:
        clean_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
    return clean_essays


def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def get_clean_essays1(essays):  # gonna => gon, na (?)
    stopWords = set(stopwords.words('english'))
    clean_essays = []
    for essay in essays:
        essay = essay.lower()
        essay = re.sub("[^a-zA-Z]", " ", essay)
        words = word_tokenize(essay)
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        clean_essays.append(wordsFiltered)
    return clean_essays


def essay_to_sent(essays):
    sent = []
    for essay in essays:
        sentence = sent_tokenize(essay)
        sent.append(sentence)
    return sent


def num_words(essays):
    num_words = []
    for essay in essays:
        num_words.append(len(essay))
    num_words = np.array(num_words)
    num_words = num_words.reshape(-1, 1)

    num_words = pd.DataFrame(num_words)
    num_words.to_csv('data/num_words.csv', sep=',')


def num_sent(essays):
    num_sent = []
    for essay in essays:
        num_sent.append(len(essay))
    num_sent = np.array(num_sent)
    num_sent = num_sent.reshape(-1, 1)

    num_sent = pd.DataFrame(num_sent)
    num_sent.to_csv('data/num_sent.csv', sep=',')
    return num_sent


csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)
Text = csv_input['essay']
essays = np.array(get_clean_essays(Text))
# print(essays)
num_words(essays)

# sent = np.array(essay_to_sent(Text))
# num_sent(sent)


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


