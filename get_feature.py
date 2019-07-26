import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
dataframe = pd.read_csv('essays_and_scores.csv', encoding='latin-1')
data = dataframe[['essay_set', 'essay', 'domain1_score']].copy()


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


features_set1 = extract_features(data[data['essay_set'] == 1])

print(features_set1)
print(type(features_set1))
features_set1 = pd.DataFrame(features_set1)
features_set1.to_csv('data/essay_set1_feature.tsv', sep='\t')
