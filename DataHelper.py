import codecs
import mysql.connector
import os
import CleanUtils as cc
import re

path = '/home/qiaoyang/bishe/SourceCodeClassify/data/'
tagSet = ['opengl','sockets','sorting', 'mfc','lambda', 'random','math','io', 'openmp','xcode',
          'arduino','jni','mingw','tree','directx','time', 'openssl','network','hash','mysql',
          'heap', 'gtk', 'graph']

def create_dict():
    tagDict = dict()
    i =0
    for tag in tagSet:
        tagDict[tag]=i
        i+=1
    return tagDict

def prepare_data():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    cursor.execute('select * from selectTagType')
    numRows = int(cursor.rowcount)

    code_train = codecs.open(path+'trainCode.txt', 'w+', 'utf8')
    type_train = codecs.open(path+'trainType.txt', 'w+', 'utf8')
    tag_train = codecs.open(path +'trainTag.txt', 'w+', 'utf8')

    code_test = codecs.open(path + 'testCode.txt', 'w+', 'utf8')
    type_test = codecs.open(path + 'testType.txt', 'w+', 'utf8')
    tag_test = codecs.open(path + 'testTag.txt', 'w+', 'utf8')

    for i in range(numRows):
        row = cursor.fetchone()
        id = row[0]
        code = cc.code_anonymous(cc.get_normalize_code(cc.remove_non_ascii_1(row[1].encode('utf-8')).replace("\n"," "),1000))
        patternBlank = re.compile(' +')
        code = re.sub(patternBlank, " ", code)
        type = str(row[2]).replace('\n','')
        tag = str(row[3]).replace('\n','')
        if(i<numRows*0.8):
            code_train.write(code+'\n')
            type_train.write(type+'\n')
            tag_train.write(tag+'\n')
        else:
            code_test.write(code + '\n')
            type_test.write(type + '\n')
            tag_test.write(tag + '\n')
    code_train.close()
    type_train.close()
    tag_train.close()
    code_test.close()
    type_test.close()
    tag_test.close()


def prepare_csv():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    cursor.execute('select * from selectTagType')
    numRows = int(cursor.rowcount)
    tagDict = create_dict()

    code_train = codecs.open(path+'train.txt', 'w+', 'utf8')
    code_dev = codecs.open(path + 'dev.txt', 'w+', 'utf8')
    code_test = codecs.open(path + 'test.txt', 'w+', 'utf8')

    for i in range(numRows):
        row = cursor.fetchone()
        id = row[0]
        code = cc.code_anonymous(cc.get_normalize_code(cc.remove_non_ascii_1(row[1].encode('utf-8')).replace("\n"," "),200))
        patternBlank = re.compile(' +')
        code = re.sub(patternBlank, " ", code).replace("@","")
        type = str(row[2]).replace('\n','')
        tag = str(tagDict.get(str(row[3]).replace('\n','')))
        if(i<numRows*0.7):
            code_train.write(tag+"@"+code+"@"+type+"\n")
        elif(i<numRows*0.8):
            code_dev.write(tag+"@"+code+"@"+type+"\n")
        else:
            code_test.write(tag+"@"+code+"@"+type+"\n")

    code_train.close()
    code_dev.close()
    code_test.close()


import re
import csv
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from keras.utils.np_utils import to_categorical
import sys
csv.field_size_limit(sys.maxsize)
# data_path = "/home/qiaoyang/codeData/binary_code/data/small_sample/"
# model_dir = '/home/qiaoyang/pythonProject/BinaryCompileClassification/models/'


# process labels
def label_to_categorical(labels, need_to_categorical):
    label = sorted(list(set(labels)))
    num_labels = len(label)
    print('label total count is: ' + str(num_labels))
    label_indict = range(num_labels)
    labels_index = dict(zip(label, label_indict))
    labels = [labels_index[y] for y in labels]
    if (need_to_categorical):
        labels = to_categorical(np.asarray(labels))
    print('Shape of label tensor:', labels.shape)
    return labels, label,labels_index


def prepare_classification_data(data_path,is_ast):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['tag', 'code','ast']
    df.columns = selected
    if(is_ast):
        texts = df[selected[2]].values.astype('U')
    else:
        texts = df[selected[1]]
        #texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]].tolist()
    print texts[0]
    return texts,labels


def prepare_dl_data(data_path):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['tag', 'code','ast']
    df.columns = selected

    code = df[selected[1]].values.astype('U')
    ast = df[selected[2]].values.astype('U')
    labels = df[selected[0]].tolist()
    return code,ast,labels


def save_obj(obj,path, name):
    with open(path+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path,name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_tokenizer(all_text,max_word,voca_path):

    texts = []
    for text in all_text:
        if(isinstance(text, basestring)):
            temp = cc.remove_blank(cc._WORD_SPLIT.split(text))
            texts.extend(temp)
    counts = Counter(texts)
    common_list = counts.most_common()
    common_list.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in common_list]
    word_picked = ['<unknown>']
    word_picked.extend(sorted_voc)
    if(len(word_picked)>max_word):
        word_picked = word_picked[:max_word]
    word_index = dict()
    for word,index in zip(word_picked,range(max_word)):
        word_index[word] = index+1
    save_obj(word_index,voca_path,'voca')
    print "unknown word index is "+str(word_index.get('<unknown>'))
    print "Nuber of unique token is "+str(len(word_index))
    return word_index



if __name__ == "__main__":
    prepare_csv()