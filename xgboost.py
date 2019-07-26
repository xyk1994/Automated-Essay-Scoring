import keras
from keras.models import Model, Sequential, load_model, model_from_config
from keras.layers import Input, concatenate,InputSpec, Activation, Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Bidirectional
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import nltk, re, collections
from collections import defaultdict
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
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
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

pred = pd.read_csv(filepath_or_buffer="data/xie7.22/predict_score.tsv", delimiter="\t""\n", engine="python", error_bad_lines=False)
pred = np.array(pred)

print(pred)

Pred = []
for i in pred:
    Pred.append(float(i))
print(Pred)