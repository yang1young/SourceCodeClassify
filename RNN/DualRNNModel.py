import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, LSTM
from keras.layers import Dense, Input, Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Utils import DataHelper as data_helper
from Utils import ModelUtils as mu
from data_prepare import StackData
from data_prepare import MouliliData
from Utils import CleanUtils as cu

MAX_SENT_LENGTH = 200
NUM_CLASS = 104
MAX_NB_WORDS = 500
EMBEDDING_DIM = 300
MAX_EPOCH = 20

def data_transfer(word_index,x,z,y):
    def data_private(word_index,x):
        data = np.zeros((len(x), MAX_SENT_LENGTH), dtype='int32')
        for i, sentences in enumerate(x):
            wordTokens = cu._WORD_SPLIT.split(sentences)
            wordTokens = cu.remove_blank(wordTokens)
            k = 0
            for _, word in enumerate(wordTokens):
                if (k < MAX_SENT_LENGTH):
                    if (word not in word_index):
                        data[i, k] = word_index['<unknown>']
                    else:
                        data[i, k] = word_index[word]
                k = k + 1
        return data
    train_code = data_private(word_index,x)
    train_ast = data_private(word_index,z)
    labels = to_categorical(np.asarray(y),num_classes=NUM_CLASS)
    print('Shape of code tensor:', train_code.shape)
    print('Shape of ast tensor:', train_ast.shape)
    print('Shape of label tensor:', labels.shape)
    return train_code,train_ast,labels



def rnn_model():

    model = Sequential()
    model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS+1, input_length=MAX_SENT_LENGTH,trainable = True,mask_zero=True))
    #model.add(Masking(mask_value=0))
    #model.add(Dropout(0.5))
    initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    #model.add(LSTM(200,kernel_initializer =initial,dropout=0.8,return_sequences=True).supports_masking)
    model.add(LSTM(200,kernel_initializer =initial,dropout=0.5,return_sequences=True))
    model.add(LSTM(100,kernel_initializer =initial,dropout=0.5))
    model.add(Dense(50))
    return model

def dual_model(model1,model2):
    merged = Merge([model1, model2], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(NUM_CLASS, activation='softmax'))
    # final_model.add(Activation('sigmoid'))
    return final_model


def cnn_model():
    convs = []
    filter_sizes = [3, 4, 5]
    sequence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SENT_LENGTH,trainable = True)(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_dropout = Dropout(0.5)(l_merge)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_dropout)
    l_bn = BatchNormalization()(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_bn)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(NUM_CLASS, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    print model.summary()
    return model


def eval_model(model,x1,x2,y,label,model_path):

    predict = model.predict([x1,x2])
    print predict
    assert len(predict)==len(y)
    y_predict = []
    y_real = []
    for p, r in zip(predict, y):
        y_predict.append(np.argmax(p))
        y_real.append(np.argmax(r))

    result = mu.precision_recall(label, y_predict, y_real)
    y_predict = np.asarray(y_predict)
    y_real = np.asarray(y_real)

    cross_table = 'crosstab:{0}'.format(pd.crosstab(y_real, y_predict, margins=True))
    precision = 'precision:{0}'.format(precision_score(y_real, y_predict, average='macro'))
    recall = 'recall:{0}'.format(recall_score(y_real, y_predict, average='macro'))
    accuracy = 'accuracy:{0}'.format(accuracy_score(y_real, y_predict))

    file = open(model_path + 'train.log', 'w')
    file.write(result + '\n' + cross_table + '\n' + precision + '\n' + recall + '\n' + accuracy + '\n')
    print result + '\n' + cross_table + '\n' + precision + '\n' + recall + '\n' + accuracy + '\n'


def train(x1_train, x2_train,y_train,x1_val, x2_val,y_val,model_path):
    model1 = rnn_model()
    model2 = rnn_model()
    model = dual_model(model1,model2)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model fitting - dual input  LSTM")
    print model.summary()
    # checkpoint
    filepath = model_path+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit([x1_train,x2_train], y_train, validation_data=([x1_val,x2_val],y_val),
              epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)

    print(history.history)
    #data_helper.save_obj(history.history, model_path, "train.log")
    return model

def re_train(x1_train, x2_train,y_train,x1_val, x2_val,y_val,model):
    filepath = model_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit([x1_train, x2_train], y_train, validation_data=([x1_val, x2_val], y_val),
                        epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)

    print(history.history)
    # data_helper.save_obj(history.history, model_path, "train.log")
    return model

def reload_model(model_path,model_name):
    model = rnn_model()
    # load weights
    model.load_weights(model_path + model_name)
    # Compile model (required to make predictions)
    #optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("Created model and loaded weights from file")
    return model



if __name__ == "__main__":

    path = "/home/qiaoyang/bishe/SourceCodeClassify/"
    data_path = path+"data/"
    train_path = data_path+"train.txt"
    dev_path = data_path+"dev.txt"
    test_path = data_path+"test.txt"
    model_path = path+'model/'

    train_code,train_ast,train_labels = data_helper.prepare_dl_data(train_path)
    dev_code,dev_ast, dev_labels = data_helper.prepare_dl_data(dev_path)
    test_code,test_ast, test_labels = data_helper.prepare_dl_data(test_path)

    df = open(train_path,'r').readlines()
    word_index = data_helper.get_tokenizer(df, MAX_NB_WORDS, model_path)

    print('Total %s unique tokens.' % len(word_index))

    x1_train,x2_train, y_train = data_transfer(word_index, train_code, train_ast,train_labels)
    x1_val, x2_val,y_val = data_transfer(word_index, dev_code,dev_ast, dev_labels)
    x1_test, x2_test,y_test = data_transfer(word_index, test_code,test_ast, test_labels)

    label = StackData.tagSet
    # label = MouliliData.tagSet

    model = train(x1_train,x2_train,y_train,x1_val,x2_val,y_val,model_path)
    # model = reload_model(model_path,'weights-improvement-14-0.63.hdf5')
    #re_train(x1_train,x2_train,y_train,x1_val,x2_val,y_val,model)
    eval_model(model,x1_test,x2_test,y_test,label,model_path)
    # #eval_model(model,x_test_d,y_test_d)