# -*-coding:utf-8-*-

import os
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Lambda, Embedding, Dropout, Bidirectional, LSTM, Average, Dense, Activation
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
import keras.backend as K

from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import random as rn

import shutil
import json

NUM = 1
MBTI = "EI"
USER_COUNT_TWEET = 200
LEN_TWEET = 140

SEED = 1
EMBEDDING_SIZE = 100
HIDDEN_LAYER_SIZE = 64
DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NUM_EPOCHS = 10
CLASS_WEIGHT = None#'auto'

READ_DICT_FILE_00 = "../DATA/08.TRAIN_DICT/dict_info.json"
READ_DICT_FILE_01 = "../DATA/08.TRAIN_DICT/dict_tweet.json"
DICT_LIST = [READ_DICT_FILE_00, READ_DICT_FILE_01]

READ_TRAIN_MBTI_FILE = "../DATA/09.DATASET_TDT/train_mbti.json"
READ_TRAIN_TWEET_FILE = "../DATA/09.DATASET_TDT/train_tweet_index_list.json"
TRAIN_LIST = [READ_TRAIN_MBTI_FILE, READ_TRAIN_TWEET_FILE]
READ_DEV_MBTI_FILE = "../DATA/09.DATASET_TDT/dev_mbti.json"
READ_DEV_TWEET_FILE = "../DATA/09.DATASET_TDT/dev_tweet_index_list.json"
DEV_LIST = [READ_DEV_MBTI_FILE, READ_DEV_TWEET_FILE]

WRITE_DIR = "../DATA/11.MODEL4/"
if os.path.exists(WRITE_DIR) == False:
    os.mkdir(WRITE_DIR)
WRITE_LOG_DIR = WRITE_DIR+"log/"+str(MBTI)+"_"+str(NUM).zfill(2)+"/"
if os.path.exists(WRITE_LOG_DIR) == False:
    os.mkdir(WRITE_LOG_DIR)
WRITE_MODEL_PICTURE_FILE = WRITE_DIR + "model_picture/"+str(MBTI)+"_"+str(NUM).zfill(2)+".png"
WRITE_MODEL_FILE = WRITE_DIR + "save_model/"+str(MBTI)+"_"+str(NUM).zfill(2)+".json"
WRITE_SAVE_WEIGHT_DIR = WRITE_DIR + "save_weight/"+str(MBTI)+"_"+str(NUM).zfill(2)+"/"
if os.path.exists(WRITE_SAVE_WEIGHT_DIR) == False:
    os.mkdir(WRITE_SAVE_WEIGHT_DIR)
WRITE_WEIGHT_BEST_FILE = WRITE_DIR + "best_weight/"+str(MBTI)+"_"+str(NUM).zfill(2)+".hdf5"

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
def make_feature_vector(read_file_list, num_data, mbti, user_count_tweet, len_tweet):
    x = np.empty((num_data, ), dtype = list)
    y = np.zeros((num_data, ))
    for i in range(num_data):
        x[i] = []
    for read_file in read_file_list:
        with open(read_file, "r") as fread:
            read_dict = json.load(fread)
        if "mbti" in read_file:
            for i, user_id in enumerate(list(read_dict)):
                y[i] = read_dict[user_id]['mbti_'+str(mbti)]
        if "tweet" in read_file:
            for i, user_id in enumerate(list(read_dict)):
                x_2 = np.empty((user_count_tweet, ), dtype = list)
                tweet_index_list = read_dict[user_id]
                for j, tweet_index in enumerate(tweet_index_list):
                    x_2[j] = [w if w != -1 else 1 for w in tweet_index]
                x[i] = sequence.pad_sequences(x_2, maxlen = len_tweet)

    x = sequence.pad_sequences(x, maxlen = user_count_tweet)
    return x, y

#モデル作成
def make_model(max_features, user_count_tweet, len_tweet, embedding_size, hidden_layer_size, dropout_rate):
    
    slicelayer = []
    embeddinglayer = []
    blstmlayer = []
    for i in range(user_count_tweet):
        slicelayer.append(Lambda(lambda x: x[:, i]))
        embeddinglayer.append(Embedding(max_features, embedding_size, input_length = len_tweet))
        blstmlayer.append(Bidirectional(LSTM(hidden_layer_size), merge_mode = 'concat', input_shape = (len_tweet, embedding_size)))
    averagelayer = Average()
    dropoutlayer = Dropout(dropout_rate)
    denselayer = Dense(1)
    activationlayer = Activation("sigmoid")
    
    x = Input(shape = (user_count_tweet, len_tweet))
    blstmlist = []
    for i in range(user_count_tweet):
        line = slicelayer[i](x)
        emb = embeddinglayer[i](line)
        blstm = blstmlayer[i](emb)
        blstmlist.append(blstm)
    ave = averagelayer(blstmlist)
    drop = dropoutlayer(ave)
    den = denselayer(drop)
    y = activationlayer(den)
    
    model = Model(inputs = x, outputs = y)
    return model                                                                                                                                

#混合行列のための計算
def print_confusion_matrix(xdev, ydev, model, user_count_tweet, len_tweet):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(xdev)):
        xdata = xdev[i].reshape(1, user_count_tweet, len_tweet)
        ylabel = int(ydev[i])
        ypred = model.predict(xdata)[0][0]
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
def print_roc_auc(xdev, ydev, model, user_count_tweet, len_tweet):

    ypredlist = []
    for i in range(len(xdev)):
        xdata = xdev[i].reshape(1, user_count_tweet, len_tweet)
        ypred = model.predict(xdata)[0][0]
        ypredlist.append(ypred)
    ypredlist = np.array(ypredlist)
    fpr, tpr, thresholds = roc_curve(ydev, ypredlist)
    AUC = auc(fpr, tpr)
    print("AUC"+"\t\t"+"= "+str(AUC)+"\n")

    return AUC

    
#メインプログラム
if __name__ == '__main__':

    #シードの固定
    seed(SEED)
    print(SEED)
    
    #辞書データ
    max_features = info_dict(DICT_LIST)
    print(max_features)
    
    #学習データ
    num_train_data = info_num_data(READ_TRAIN_TWEET_FILE)
    num_dev_data = info_num_data(READ_DEV_TWEET_FILE)
    print(num_train_data)
    print(num_dev_data)

    #学習データとテストデータを特徴ベクトルに変換
    xtrain, ytrain = make_feature_vector(TRAIN_LIST, num_train_data, MBTI, USER_COUNT_TWEET, LEN_TWEET)
    xdev, ydev = make_feature_vector(DEV_LIST, num_dev_data, MBTI, USER_COUNT_TWEET, LEN_TWEET)
    print(xtrain.shape, ytrain.shape, xdev.shape, ydev.shape)

    #モデルの作成
    model = make_model(max_features, USER_COUNT_TWEET, LEN_TWEET, EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, DROPOUT_RATE)
        
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()
    plot_model(model, to_file = WRITE_MODEL_PICTURE_FILE, show_shapes = True)
    history = model.fit(xtrain,
                        ytrain,
                        batch_size = BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        verbose = 2,
                        callbacks = [TensorBoard(WRITE_LOG_DIR),
                                     ModelCheckpoint(WRITE_SAVE_WEIGHT_DIR+"{epoch:02d}.hdf5",save_weights_only=True)],
                        validation_data = (xdev, ydev),
                        class_weight = CLASS_WEIGHT)
    with open(WRITE_MODEL_FILE, "w") as fmodel:
        fmodel.write(model.to_json())
    max_AUC = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        print("epoch:"+str(epoch).zfill(2))
        model.load_weights(WRITE_SAVE_WEIGHT_DIR+str(epoch).zfill(2)+".hdf5")

        #混合行列出力
        print_confusion_matrix(xdev, ydev, model, USER_COUNT_TWEET, LEN_TWEET)

        #ROC出力
        AUC = print_roc_auc(xdev, ydev, model, USER_COUNT_TWEET, LEN_TWEET)

        if max_AUC < AUC:
            max_AUC = AUC
            max_epoch = epoch

    print("***BEST_EPOCH***")
    print("epoch\t\t= "+str(max_epoch).zfill(2))
    print("AUC\t\t= "+str(max_AUC))
    shutil.copyfile(WRITE_SAVE_WEIGHT_DIR+str(max_epoch).zfill(2)+".hdf5", WRITE_WEIGHT_BEST_FILE)
