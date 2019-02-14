# -*-coding:utf-8-*-
#embedding

import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import sequence

from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K

import tensorflow as tf
import random as rn

import json

SEED = 1
NUM = 1
MAX_SENTENCE_LENGTH = 15000

NUM_SAMPLE = 10

READ_DICT_FILE_00 = "../DATA/08.TRAIN_DICT/dict_00.json"
READ_DICT_FILE_01 = "../DATA/08.TRAIN_DICT/dict_01.json"
DICT_LIST = [READ_DICT_FILE_00, READ_DICT_FILE_01]

READ_TEST_MBTI_FILE = "../DATA/09.DATASET_TDT/test_mbti.json"
READ_TEST_TWEET_FILE = "../DATA/09.DATASET_TDT/test_tweet_word_binary.json"
TEST_LIST = [READ_TEST_MBTI_FILE, READ_TEST_TWEET_FILE]

READ_MODEL_FILE = "../DATA/11.MODEL5/save_model/"+str(NUM).zfill(2)+".json"
READ_WEIGHT_BEST_FILE = "../DATA/11.MODEL5/best_weight/"+str(NUM).zfill(2)+".hdf5"

WRITE_ROC_PICTURE_FILE = "../DATA/11.MODEL5/test_roc_picture/"+str(NUM).zfill(2)+".png"

#シードの固定
def seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
    tf.set_random_seed(seed)
    session = tf.Session(graph = tf.get_default_graph(), config = session_conf)
    K.set_session(session)

#辞書情報
def info_dict(dict_list):
    index_dict = {}
    max_features = 0
    for read_file in dict_list:
        with open(read_file, "r") as fread:
            read_dict = json.load(fread)
        for key in list(read_dict):
            index_dict[key] = int(read_dict[key]) + max_features
        max_features = max(index_dict.values()) + 1
    return max_features

#学習データ情報
def info_num_data(read_file):
    with open(read_file, "r") as fread:
        read_dict = json.load(fread)
    num_data = len(list(read_dict))
    return num_data

#特徴ベクトル作成
def make_feature_vector(read_file_list, num_data, max_sentence_length):
    x = np.empty((num_data, ), dtype = list)
    y_EI = np.zeros((num_data, ))
    y_NS = np.zeros((num_data, ))
    y_TF = np.zeros((num_data, ))
    y_JP = np.zeros((num_data, ))
    for i in range(num_data):
        x[i] = []
    for read_file in read_file_list:
        with open(read_file, "r") as fread:
            read_dict = json.load(fread)
        if "mbti" in read_file:
            for i, user_id in enumerate(list(read_dict)):
                y_EI[i] = read_dict[user_id]['mbti_EI']
                y_NS[i] = read_dict[user_id]['mbti_NS']
                y_TF[i] = read_dict[user_id]['mbti_TF']
                y_JP[i] = read_dict[user_id]['mbti_JP']
        if "tweet" in read_file:
            for i, user_id in enumerate(list(read_dict)):
                x[i] += read_dict[user_id]

    x = sequence.pad_sequences(x, maxlen = max_sentence_length)
    return x, y_EI, y_NS, y_TF, y_JP

def print_sample(xtest, ytest, model, max_sentence_length, num_sample):
    for i in range(num_sample):
        num_random = np.random.randint(len(xtest))
        xdata = xtest[num_random].reshape(1, max_sentence_length)
        ylabel = ytest[num_random]
        ypred = model.predict(xdata)[0][0]
        print(str(ypred)+"\t"+str(ylabel)+"\n"+str(xdata))

#混合行列のための計算
def print_confusion_matrix(xtest, ytest, num_mbti, model, max_sentence_length):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(xtest)):
        xdata = xtest[i].reshape(1, max_sentence_length)
        ylabel = int(ytest[i])
        ypred = model.predict(xdata)[num_mbti][0][0]
        ypred = [1 if ypred >= 0.5 else 0][0]
        if (ylabel == ypred) and (ylabel == 1):
            tp += 1
        if (ylabel != ypred) and (ylabel == 1):
            fn += 1
        if (ylabel != ypred) and (ylabel == 0):
            fp += 1
        if (ylabel == ypred) and (ylabel == 0):
            tn += 1
    sum_class_p = tp + fn
    sum_class_n = fp + tn
    sum_pred_p = tp + fp
    sum_pred_n = fn + tn
    sum_all = tp + fn + fp + tn
    precision_p = [float(tp / sum_pred_p) if sum_pred_p != 0 else 0][0]
    precision_n = [float(tn / sum_pred_n) if sum_pred_n != 0 else 0][0]
    recall_p = [float(tp / sum_class_p) if sum_class_p != 0 else 0][0]
    recall_n = [float(tn / sum_class_n) if sum_class_n != 0 else 0][0]
    fscore_p = [float((2 * precision_p * recall_p) / (precision_p + recall_p)) if (precision_p + recall_p) != 0 else 0][0]
    fscore_n = [float((2 * precision_n * recall_n) / (precision_n + recall_n)) if (precision_n + recall_n) != 0 else 0][0]            
    fscore = float((fscore_p + fscore_n) / 2)
    if sum_class_p > sum_class_n:
        maj = sum_class_p
        majority = float(sum_class_p / sum_all)
    else:
        maj = sum_class_n
        majority = float(sum_class_n / sum_all)
    accuracy = (tp + tn) / sum_all
    
    print("***Confusion Matrix***")
    print("\t"+"pred_p"+"\t"+"pred_n"+"\t"+"sum")
    print("class_p"+"\t"+str(tp)+"\t"+str(fn)+"\t"+str(sum_class_p))
    print("class_n"+"\t"+str(fp)+"\t"+str(tn)+"\t"+str(sum_class_n))
    print("sum"+"\t"+str(sum_pred_p)+"\t"+str(sum_pred_n)+"\t"+str(sum_all))
    print("***Score***")
    print("Precision_p"+"\t"+"= "+str(precision_p)+" ("+str(tp)+"/"+str(sum_pred_p)+")")
    print("Precision_n"+"\t"+"= "+str(precision_n)+" ("+str(tn)+"/"+str(sum_pred_n)+")")
    print("Recall_p"+"\t"+"= "+str(recall_p)+" ("+str(tp)+"/"+str(sum_class_p)+")")
    print("Recall_n"+"\t"+"= "+str(recall_n)+" ("+str(tn)+"/"+str(sum_class_n)+")")
    print("F-score_p"+"\t"+"= "+str(fscore_p))
    print("F-score_n"+"\t"+"= "+str(fscore_n))
    print("F-score"+"\t\t"+"= "+str(fscore))
    print("Majority"+"\t"+"= "+str(majority)+" ("+str(maj)+"/"+str(sum_all)+")")
    print("Accuracy"+"\t"+"= "+str(accuracy)+" ("+str(tp+tn)+"/"+str(sum_all)+")")

#ROC,AUCの出力
def print_roc_auc(xtest, ytest, num_mbti, model, max_sentence_length, roc_picture_file):

    ypredlist = []
    for i in range(len(xtest)):
        xdata = xtest[i].reshape(1, max_sentence_length)
        ypred = model.predict(xdata)[num_mbti][0][0]
        ypredlist.append(ypred)
    ypredlist = np.array(ypredlist)
    fpr, tpr, thresholds = roc_curve(ytest, ypredlist)
    plt.plot(fpr, tpr, label = "ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize = 10, label = "thereshold zero", fillstyle = "none", c = 'k', mew = 2)
    plt.legend(loc = 4)
    plt.savefig(roc_picture_file)
    
    AUC = auc(fpr, tpr)
    print("AUC"+"\t\t"+"= "+str(AUC))


    
#メインプログラム
if __name__ == '__main__':

    #シードの固定
    seed(SEED)
    
    #辞書データ
    max_features = info_dict(DICT_LIST)
    print(max_features)

    #学習データ
    num_test_data = info_num_data(READ_TEST_TWEET_FILE)
    print(num_test_data)
    
    #学習データとテストデータを特徴ベクトルに変換 
    xtest, ytest_EI, ytest_NS, ytest_TF, ytest_JP = make_feature_vector(TEST_LIST, num_test_data, MAX_SENTENCE_LENGTH)

    #モデルの作成
    with open(READ_MODEL_FILE,"r") as fread:
        model = model_from_json(fread.read())
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()

    #重み読込み
    model.load_weights(READ_WEIGHT_BEST_FILE)
    
    ytest_list = [ytest_EI, ytest_NS, ytest_TF, ytest_JP]
    str_ytest_list = ["EI", "NS", "TF", "JP"]
    for num_mbti, ytest in enumerate(ytest_list):

        print(str_ytest_list[num_mbti])
        
        #サンプル出力
        print_sample(xtest, ytest, num_mbti, model, MAX_SENTENCE_LENGTH, NUM_SAMPLE)
        
        #混合行列出力
        print_confusion_matrix(xtest, ytest, num_mbti, model, MAX_SENTENCE_LENGTH)
        
        #ROC出力
        print_roc_auc(xtest, ytest, num_mbti, model, MAX_SENTENCE_LENGTH, WRITE_ROC_PICTURE_FILE)
