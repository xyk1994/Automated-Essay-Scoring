import keras
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


def create_dict(text):
    print("create_dict")
    word_index = {}
    for sentence in text:
        for word in sentence:
            if word not in word_index:
                word_index[word] = len(word_index)
    word_index['unk'] = len(word_index)
    return word_index


def essay_to_index(essays, word_index):
    Essays = []
    for essay in essays:
        sentence = []
        for sent in essay:
            words = []
            for word in sent:
                words.append(word_index[word])
            sentence.append(words)
        Essays.append(sentence)
    return Essays


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


csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)

Text = csv_input['essay']

Score = csv_input['domain1_score']
# Score = (Score - 2) / 10

overall_feature = csv_input[['word_count', 'sent_count', 'avg_word_len', 'lemma_count', 'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 'adv_count']]
overall_feature = np.sqrt(overall_feature)
num_feature = 9

# 做字典
clean_essays = get_clean_essays(Text)
word_index = create_dict(clean_essays)


text = np.array(Text)
# print(text)
a = []
for essay in text:
    a.append(sent_tokenize(essay))
texts = np.array(a)
# print(texts)
# [list(['Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble!', 'Thing about!', 'Dont you think so?', 'How would you feel if your teenager is always on the phone with friends!', 'Do you ever time to chat with your friends or buisness partner about things.', "Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect.", 'Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it.', 'How did you learn about other countrys/states outside of yours?', "Well I have by computer/internet, it's a new way to learn about what going on in our time!", "You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows.", 'Believe it or not the computer is much interesting then in class all day reading out of books.', "If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right.", 'You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by.', 'Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place.', 'Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble.', 'Thank you for listening.'])
#  list(['Dear @CAPS1 @CAPS2, I believe that using computers will benefit us in many ways like talking and becoming friends will others through websites like facebook and mysace.', 'Using computers can help us find coordibates, locations, and able ourselfs to millions of information.', 'Also computers will benefit us by helping with jobs as in planning a house plan and typing a @NUM1 page report for one of our jobs in less than writing it.', 'Now lets go into the wonder world of technology.', 'Using a computer will help us in life by talking or making friends on line.', 'Many people have myspace, facebooks, aim, these all benefit us by having conversations with one another.', 'Many people believe computers are bad but how can you make friends if you can never talk to them?', 'I am very fortunate for having a computer that can help with not only school work but my social life and how I make friends.', 'Computers help us with finding our locations, coordibates and millions of information online.', "If we didn't go on the internet a lot we wouldn't know how to go onto websites that @MONTH1 help us with locations and coordinates like @LOCATION1.", 'Would you rather use a computer or be in @LOCATION3.', 'When your supposed to be vacationing in @LOCATION2.', 'Million of information is found on the internet.', 'You can as almost every question and a computer will have it.', 'Would you rather easily draw up a house plan on the computers or take @NUM1 hours doing one by hand with ugly erazer marks all over it, you are garrenteed that to find a job with a drawing like that.', "Also when appling for a job many workers must write very long papers like a @NUM3 word essay on why this job fits you the most, and many people I know don't like writing @NUM3 words non-stopp for hours when it could take them I hav an a computer.", 'That is why computers we needed a lot now adays.', 'I hope this essay has impacted your descion on computers because they are great machines to work with.', 'The other day I showed my mom how to use a computer and she said it was the greatest invention sense sliced bread!', 'Now go out and buy a computer to help you chat online with friends, find locations and millions of information on one click of the button and help your self with getting a job with neat, prepared, printed work that your boss will love.'])
#  list(['Dear, @CAPS1 @CAPS2 @CAPS3 More and more people use computers, but not everyone agrees that this benefits society.', 'Those who support advances in technology believe that computers have a positive effect on people.', 'Others have different ideas.', 'A great amount in the world today are using computers, some for work and spme for the fun of it.', 'Computers is one of mans greatest accomplishments.', 'Computers are helpful in so many ways, @CAPS4, news, and live streams.', "Don't get me wrong way to much people spend time on the computer and they should be out interacting with others but who are we to tell them what to do.", 'When I grow up I want to be a author or a journalist and I know for a fact that both of those jobs involve lots of time on time on the computer, one @MONTH1 spend more time then the other but you know exactly what @CAPS5 getting at.', 'So what if some expert think people are spending to much time on the computer and not exercising, enjoying natures and interacting with family and friends.', "For all the expert knows that its how must people make a living and we don't know why people choose to use the computer for a great amount of time and to be honest it's non of my concern and it shouldn't be the so called experts concern.", 'People interact a thousand times a day on the computers.', 'Computers keep lots of kids of the streets instead of being out and causing trouble.', 'Computers helps the @ORGANIZATION1 locate most wanted criminals.', 'As you can see computers are more useful to society then you think, computers benefit society.'])
#  ...
#  list(['My opinion is that people should have computers in their homes.', 'Computers should be for important things like searching for jobs and other things.', 'People can do their work in the computer.', 'To teach your childrens how to use a computer or let their children go on the internet to they can search kids stuff.', "People shouldn't all their time on the computers and spend less time with their families and friends.", 'I think people should spend more time with their families and their friends then spending it on the computer.', 'But a lot of people use computers almost everyday.', 'But a lot of people use computers almost everyday.', 'Probably your family are worried about people spending their time everyday in the computer.', 'They their friends how to type in the computer and teach them some stuff about the history.', 'Do you think that people should alot more time with their families and be happy about it the they are spending time with their families.', "I think people shouldn't use the computer when they have a vistor over their house they don't their vistor think you are attracted to the computer.", 'They should give people the ability to learn about faraway places and to learn things about the internet.', "But not everyone agree that people shouldn't use computers because people can get attracted to it and people don't want that.", 'To happen.', 'And families are concerned about that a lot of people use this can of stuff.', 'Other people think that this is a great idea to use a lot of computer and spend more on it then their families.', 'So do you think that people should spend more time on their families and friends then picking atention to the computers.'])
#  list(['Dear readers, I think that its good and bad to use the computer to much'])
#  list(['Dear - Local Newspaper I agree thats computers are good for society.', 'Without computers a lot of things couldn?t be done.', 'Computers are sometimes the easy way out.', 'And thats why I love them.', 'Computers almost makes anything possible.', 'Now say if your an elderly person and you cant get up and your bodys really bad.', 'Well all you need is a computer.', 'You can pay your bills online or you can get a job online or even shop online.', 'All you need is a computer.', 'Computers are also good if your lazy.', 'You can just lay in bed all day and go online to work or to the mall and order things all youll have to care about is personal hygenic.', 'Sometimes computers take a lot of stress of you.', 'No more ignorant co-workers or no angry boss everythings a-okay.', 'Computers are also swel because its the eazy way out.', 'But you have to make sure you get a good computer.', 'Not and old one because it can breakdown.', 'Dats one reason why computers arent so good.', 'But as i said computers are very good they might be a little pesky but ones you get the hand of them everything gonna be alright and remember you can do almost anything with a computer.'])]

essays = []
for essay in texts:
    sentence = []
    for sents in essay:
        sent = essay_to_wordlist(sents, remove_stopwords=True)
        # sent = np.array(sent)
        sentence.append(sent)
    essays.append(sentence)
essays = np.array(essays)

# print(essays)
Essays = essay_to_index(essays, word_index)

# print(Essays[0])
# [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 6, 17, 12, 18, 19, 20, 21, 12, 22], [23], [24, 3], [25, 26, 27, 28, 29, 15], [30, 13, 14, 15, 31, 32, 33], [34, 16, 35, 14, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44], [3, 45, 46, 47, 36, 27, 48, 29, 49, 50, 51, 52, 53], [18, 54, 55, 56], [34, 36, 39, 16, 35, 18, 57, 13], [58, 3, 59, 60, 61, 13, 36, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72], [73, 36, 71, 74, 75, 76, 77, 78], [59, 79, 36, 1, 80, 81, 15, 82, 83, 84, 85, 86, 87], [58, 85, 59, 41, 88, 89, 90, 91], [92, 59, 36, 8, 93, 94, 95, 96, 97, 79, 98, 99], [100, 101, 102, 103, 104, 5, 7, 4, 59, 105, 12, 13, 14, 15, 16, 6, 17, 12, 18, 19, 73, 21, 12, 22], [106, 107]]


max_sent_length = 52
max_word_length = 86
'''
a = 0
for essay in essays:
    if max_sent_length < len(essay):
        max_sent_length = len(essay)
        a = essay
    for sent in essay:
        if max_word_length < len(sent):
            max_word_length = len(sent)
print('max sent length = ', max_sent_length)  # 52
print(a)
print('max word length = ', max_word_length)  # 86
'''

essay_pad = []
for essay in Essays:
    pad = pad_sequences(essay, maxlen=max_word_length)
    essay_pad.append(pad)

#print(essays_pad)
#print(len(essays_pad))
#print(essays_pad[0].shape)
#print(essays_pad[0][0].shape)
essays_pad = []
for essay in essay_pad:
    essays = []
    for sent in essay:
        for word in sent:
            essays.append(word)
    essays_pad.append(essays)

essays_pad = np.array(essays_pad)
# print(essays_pad)
# print(essays_pad.shape)
Padding_Length = max_word_length * max_sent_length
essays_pad = pad_sequences(essays_pad, maxlen=Padding_Length)

# print(essays_pad)
# print(essays_pad.shape)
# [[    0     0     0 ...     0   106   107]
#  [    0     0     0 ...   157    47   217]
#  [    0     0     0 ...     5   109   221]
#  ...
#  [    0     0     0 ...  5582 15200     5]
#  [    0     0     0 ...    53    36    71]
#  [    0     0     0 ...   169   446    36]]
# (1783, 4472)



embeddings_index = get_embeddings_index()
embeddings_matrix = embeddings_matrix(embeddings_index, word_index, embedding_dim=100)


def stack(tensor):
    Tensor = keras.backend.stack(tensor, axis=1)
    return Tensor


# def concat(tensor1, tensor2):
#     return Concatenate([tensor1, tensor2])


def get_model():
    print('Build model ...')
    text_input = Input(shape=(Padding_Length, ), name='text_input')
    overall_feature_input = Input(shape=(num_feature,), name='overall_feature_input')
    print(text_input)  # Tensor("text_input:0", shape=(?, 4472), dtype=float32)
    print(overall_feature_input)
    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=100,
                                weights=[embeddings_matrix],
                                input_length=Padding_Length,
                                trainable=False,
                                mask_zero=True)
    embedding = embedding_layer(text_input)  # tensor (?, 4472, 100)
    lstm = LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(embedding)  # (?, ?, 64)

    a = [i for i in range(85, 4472, 86)]
    sent_layer = [Lambda(lambda t: t[:, i, :])(lstm) for i in a]
    sent_output = Lambda(stack)(sent_layer)
    print(sent_output)  # (?, 52, 64)

    lstm2 = LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False)(sent_output)
    # dense1 = Dense(128, activation='relu')(lstm)
    print(lstm2)  # (?, 32)

    concat_layer = Lambda(concatenate)([lstm2, overall_feature_input])

    dense1 = Dense(32, activation='tanh')(concat_layer)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation='relu')(dense2)
    model = Model(inputs=[text_input, overall_feature_input], outputs=output)
    model.summary()
    return model


results = []
score_predict_list = []
count = 1
cv = KFold(n_splits=4, shuffle=False)

for train, test in cv.split(essays_pad):
    text_train, text_test = essays_pad[train], essays_pad[test]
    label_train, label_test = Score.iloc[train], Score.iloc[test]
    overall_train, overall_test = overall_feature.iloc[train], overall_feature.iloc[test]
    model = get_model()
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['mse'])

    model.fit([text_train, overall_train], label_train, batch_size=64, epochs=50)

    score_predict = model.predict([text_test, overall_test])
    score_predict = np.around(score_predict)
    result = cohen_kappa_score(label_test.values, score_predict, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))



