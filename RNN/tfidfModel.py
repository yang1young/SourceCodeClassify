#!/usr/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, accuracy_score

from utils import DataHelper


def tf_idf_model(x_train):
    # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vect = CountVectorizer(ngram_range=(1, 1), min_df=100, max_features=10000)
    X_train_counts = count_vect.fit_transform(x_train)
    print count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)
    return x_train, count_vect, tfidf_transformer


def eval_model(real, predict, name):
    assert len(predict) == len(real)
    print name
    print 'crosstab:{0}'.format(pd.crosstab(np.asarray(real), np.asarray(predict), margins=True))
    print 'precision:{0}'.format(precision_score(real, predict, average='macro'))
    print 'recall:{0}'.format(recall_score(real, predict, average='macro'))
    print 'accuracy:{0}'.format(accuracy_score(real, predict))


# def train(train_path, test_path, test_path_duplicate, is_bytecode):
#     train_x, train_y = data_helper.prepare_classification_data(train_path, is_bytecode)
#     test_x, test_y = data_helper.prepare_classification_data(test_path, is_bytecode)
#     test_x_d, test_y_d = data_helper.prepare_classification_data(test_path_duplicate, is_bytecode)
#     print len(train_y)
#     print len(train_x)
#     x_train, count_vect, tfidf_transformer = tf_idf_model(train_x)
#
#     x_test = count_vect.transform(test_x)
#     x_test = tfidf_transformer.transform(x_test)
#     x_test_d = count_vect.transform(test_x_d)
#     x_test_d = tfidf_transformer.transform(x_test_d)
#
#     # clf = MultinomialNB().fit(x_train, y_train)
#     clf = tree.DecisionTreeClassifier().fit(x_train, train_y)
#
#     predict_distict = clf.predict(x_test)
#     eval_model(test_y, predict_distict, "distinct data evaluation")
#
#     predict_duplicate = clf.predict(x_test_d)
#     eval_model(test_y_d, predict_duplicate, "contains duplicated data evaluation")
#
#     return clf, count_vect, tfidf_transformer
#

def train(train_path,test_path, is_ast):
    train_x, train_y = DataHelper.prepare_classification_data(train_path, is_ast)
    test_x, test_y = DataHelper.prepare_classification_data(test_path, is_ast)
    print len(train_y)
    print len(train_x)
    x_train, count_vect, tfidf_transformer = tf_idf_model(train_x)
    print 'finish tf idf'
    # clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(x_train, train_y)
    print 'finish training'
    del train_x,train_y,x_train
    # data_helper.save_obj(clf,path,'clf')
    # data_helper.save_obj(count_vect,path,'countvec')
    # data_helper.save_obj(tfidf_transformer,path,'tfidf')
    x_test = count_vect.transform(test_x)
    x_test = tfidf_transformer.transform(x_test)
    print 'finish test data'
    predict_distict = clf.predict(x_test)
    print len(predict_distict)
    eval_model(test_y, predict_distict, "distinct data evaluation")
    return clf, count_vect, tfidf_transformer


if __name__ == "__main__":
    path = "/home/qiaoyang/bishe/SourceCodeClassify/data/"
    #path = path+"train_repalce_number/"
    train_path = path+"train.txt"
    test_path = path+"test.txt"
    #test_path_duplicate = "/home/qiaoyang/codeData/binary_code/newData/data.dev"
    is_ast = False
    clf, count_vect, tfidf_transformer = train(train_path, test_path,is_ast)
    # clf = data_helper.load_obj(path,'clf')
    # count_vect = data_helper.load_obj(path,'countvec')
    # tfidf_transformer = data_helper.load_obj(path,'tfidf')
    # predict(clf, count_vect, tfidf_transformer, test_path, is_bytecode)
    #ensamble_test(data_path, clf, 0.2, count_vect, tfidf_transformer, is_bytecode)