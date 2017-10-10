from __future__ import print_function

from keras.layers import Dense, Merge
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import os
import numpy as np
import codecs
np.random.seed(1337)
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence  import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
import csv
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

DATADIR = '/home/qiaoyang/bishe/SourceCodeClassify/data'

MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.1
CLASS_NUMBER =84
# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4
lstm_output_size = 150

def getTrain(file_name,need_label,needCSV):

    if (needCSV):
        df = pd.read_csv(DATADIR + file_name, sep='@', header=None, encoding='utf8', engine='python')
        selected = ['Code', 'Tag']
        df.columns = selected
        texts = (df[selected[0]]).tolist()
        texts = [s.encode('utf-8') for s in texts]
        if (need_label):
            label = sorted(list(set(df[selected[1]].tolist())))
            num_labels = len(label)
            print(label)
            lableIndict = range(num_labels)
            labels_index = dict(zip(label, lableIndict))
            labels = df[selected[1]].apply(lambda y: labels_index[y]).tolist()
            return texts, labels, label
        else:
            return texts
    else:
        df = pd.read_csv(DATADIR + 'tags.csv', sep='@', header=None, encoding='utf8', engine='python')
        selected = ['Tag']
        df.columns = selected
        texts = open(DATADIR + file_name, 'r')
        texts = [s.encode('utf-8') for s in texts]

        if (need_label):
            label = sorted(list(set(df[selected[0]].tolist())))
            num_labels = len(label)
            lableIndict = range(num_labels)
            labels_index = dict(zip(label, lableIndict))
            labels = df[selected[0]].apply(lambda y: labels_index[y]).tolist()
            return texts, labels, label
        else:
            return texts


def get_token(texts):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data,word_index


def dataPrepare(needCSV):
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'code.vectors.glove.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    source_data,labels,label =getTrain('replaceData.csv',True,needCSV)
    type_data= getTrain('typeData.csv',False,needCSV)

    source_data_token,source_word_index = get_token(source_data)
    type_data_token,type_word_index = get_token(type_data)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', source_data_token.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(source_data_token.shape[0])
    np.random.shuffle(indices)
    source_data_token = source_data_token[indices]
    type_data_token = type_data_token[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * source_data_token.shape[0])*2

    source_train = source_data_token[:-nb_validation_samples]
    source_test =source_data_token[-nb_validation_samples:]
    type_train =type_data_token[:-nb_validation_samples]
    type_test =type_data_token[-nb_validation_samples:]
    label_train = labels[:-nb_validation_samples]
    label_test = labels[-nb_validation_samples:]


    return source_train,source_test,type_train,type_test,\
           label_train,label_test,source_word_index,type_word_index,embeddings_index,label



def get_model(word_index,embeddings_index):

    print('Preparing embedding matrix.')

    # num_words = min(MAX_NB_WORDS, len(word_index))
    num_words =20000
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        # if i>=word_index.__len__():
        #     break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Training model.')
    model1 = Sequential()
    model1.add(embedding_layer)
    model1.add(Dropout(0.25))
    model1.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model1.add(MaxPooling1D(pool_length=pool_length))
    model1.add(LSTM(lstm_output_size))
    model1.add(Dense(CLASS_NUMBER))
    return model1


def train(needCSV):
    source_train, source_test,  type_train, type_test, \
    label_train, label_test, source_word_index,  type_word_index, embeddings_index, label = dataPrepare(needCSV)

    model1 = get_model(source_word_index,embeddings_index)
    print(source_word_index)
    model3 = get_model(type_word_index,embeddings_index)

    merged = Merge([model1,model3], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(CLASS_NUMBER, activation='softmax'))
    #final_model.add(Activation('sigmoid'))

    final_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    final_model.fit([source_train, type_train],label_train,nb_epoch=50, batch_size=128)
    predict = final_model.predict([source_test,type_test])

    countSet = []
    predictSet = []
    rightSet = []
    for i in range(len(label)):
        countSet.append(0.0)
        predictSet.append(0.0)
        rightSet.append(0.0)

    for p, r in zip(predict, label_test):
        indexP = np.argmax(p)
        indexR = np.argmax(r)
        # name = label[int(indexR)]
        countSet[int(indexR)] += 1
        predictSet[int(indexP)] += 1
        if (int(indexR) == int(indexP)):
            rightSet[int(indexP)] += 1
    count = 0
    result_file = open(DATADIR+'dualResult.csv','w')
    for i,j,k in zip(countSet,predictSet,rightSet):
        tag = str(label[count])
        if(i==0):
            recall = '0'
        else:
            recall = str(k/i)
        if(j ==0):
            precision = '0'
        else:
            precision = str(k/j)
        print(tag+','+ str(i) +','+str(precision)+','+str(recall))
        result_file.write(tag+','+ str(i) +','+str(precision)+','+str(recall)+'\n')
        count +=1




#train()
