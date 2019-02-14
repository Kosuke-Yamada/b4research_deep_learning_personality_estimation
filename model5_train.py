# -*-coding:utf-8-*-

import os
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Lambda, Dropout, Dense, Activation
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
MAX_SENTENCE_LENGTH = 15000

SEED = 1
EMBEDDING_SIZE = 100
HIDDEN_LAYER_SIZE = 64
DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NUM_EPOCHS = 100
CLASS_WEIGHT = None#'auto'

READ_DICT_FILE_00 = "../DATA/08.TRAIN_DICT/dict_00.json"
READ_DICT_FILE_01 = "../DATA/08.TRAIN_DICT/dict_01.json"
DICT_LIST = [READ_DICT_FILE_00, READ_DICT_FILE_01]

READ_TRAIN_MBTI_FILE = "../DATA/09.DATASET_TDT/train_mbti.json"
READ_TRAIN_TWEET_FILE = "../DATA/09.DATASET_TDT/train_tweet_index.json"
TRAIN_LIST = [READ_TRAIN_MBTI_FILE, READ_TRAIN_TWEET_FILE]
READ_DEV_MBTI_FILE = "../DATA/09.DATASET_TDT/dev_mbti.json"
READ_DEV_TWEET_FILE = "../DATA/09.DATASET_TDT/dev_tweet_index.json"
DEV_LIST = [READ_DEV_MBTI_FILE, READ_DEV_TWEET_FILE]

WRITE_DIR = "../DATA/11.MODEL5/"
if os.path.exists(WRITE_DIR) == False:
    os.mkdir(WRITE_DIR)
WRITE_LOG_DIR = WRITE_DIR+"log/"+str(NUM).zfill(2)+"/"
if os.path.exists(WRITE_LOG_DIR) == False:
    os.mkdir(WRITE_LOG_DIR)
WRITE_MODEL_PICTURE_FILE = WRITE_DIR + "model_picture/"+str(NUM).zfill(2)+".png"
WRITE_MODEL_FILE = WRITE_DIR + "save_model/"+str(NUM).zfill(2)+".json"
WRITE_SAVE_WEIGHT_DIR = WRITE_DIR + "save_weight/"+str(NUM).zfill(2)+"/"
if os.path.exists(WRITE_SAVE_WEIGHT_DIR) == False:
    os.mkdir(WRITE_SAVE_WEIGHT_DIR)
WRITE_WEIGHT_BEST_DIR = WRITE_DIR + "best_weight/"

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
                x[i] += [w if w != -1 else 1 for w in read_dict[user_id]]

    x = sequence.pad_sequences(x, maxlen = max_sentence_length)
    return x, y_EI, y_NS, y_TF, y_JP

#モデル作成
def make_model(max_features, embedding_size, max_sentence_length, dropout_rate, hidden_layer_size):
    
    embeddinglayer = Embedding(max_features, embedding_size, input_length = max_sentence_length)
    meanlayer = Lambda(lambda x: K.mean(x, axis=1), output_shape = (embedding_size, ))
    
    dropoutlayer = []
    denselayer = []
    activationlayer = []
    for i in range(4):
        dropoutlayer.append(Dropout(dropout_rate))
        denselayer.append(Dense(1))
        activationlayer.append(Activation("sigmoid"))
        
    x = Input(shape = (max_sentence_length, ))
    emb = embeddinglayer(x)
    mean = meanlayer(emb)
    for i in range(4):
        if i == 0:
            drop = dropoutlayer[i](mean)
            den = denselayer[i](drop)
            y_EI = activationlayer[i](den)
        elif i == 1:
            drop = dropoutlayer[i](mean)
            den = denselayer[i](drop)
            y_NS = activationlayer[i](den)
        elif i == 2:
            drop = dropoutlayer[i](mean)
            den = denselayer[i](drop)
            y_TF = activationlayer[i](den)
        elif i == 3:
            drop = dropoutlayer[i](mean)
            den = denselayer[i](drop)
            y_JP = activationlayer[i](den)
            
    model = Model(inputs = x, outputs = [y_EI, y_NS, y_TF, y_JP])
    return model

#混合行列のための計算
def print_confusion_matrix(xdev, ydev, num_mbti, model, max_sentence_length):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(xdev)):
        xdata = xdev[i].reshape(1, max_sentence_length)
        ylabel = int(ydev[i])
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
def print_roc_auc(xdev, ydev, num_mbti, model, max_sentence_length):

    ypredlist = []
    for i in range(len(xdev)):
        xdata = xdev[i].reshape(1, max_sentence_length)
        ypred = model.predict(xdata)[num_mbti][0][0]
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
    xtrain, ytrain_EI, ytrain_NS, ytrain_TF, ytrain_JP = make_feature_vector(TRAIN_LIST, num_train_data, MAX_SENTENCE_LENGTH)
    xdev, ydev_EI, ydev_NS, ydev_TF, ydev_JP = make_feature_vector(DEV_LIST, num_dev_data, MAX_SENTENCE_LENGTH)
    print(xtrain.shape, ytrain_EI.shape, xdev.shape, ydev_EI.shape)

    #モデルの作成
    model = make_model(max_features, EMBEDDING_SIZE, MAX_SENTENCE_LENGTH, DROPOUT_RATE, HIDDEN_LAYER_SIZE)
        
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()
    plot_model(model, to_file = WRITE_MODEL_PICTURE_FILE, show_shapes = True)
    history = model.fit(xtrain,
                        [ytrain_EI, ytrain_NS, ytrain_TF, ytrain_JP],
                        batch_size = BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        verbose = 2,
                        callbacks = [TensorBoard(WRITE_LOG_DIR),
                                     ModelCheckpoint(WRITE_SAVE_WEIGHT_DIR+"{epoch:02d}.hdf5")],
                        validation_data = (xdev, [ydev_EI, ydev_NS, ydev_TF, ydev_JP]),
                        class_weight = CLASS_WEIGHT)
    with open(WRITE_MODEL_FILE, "w") as fmodel:
        fmodel.write(model.to_json())

    ydev_list = [ydev_EI, ydev_NS, ydev_TF, ydev_JP]
    str_ydev_list = ["EI", "NS", "TF", "JP"]
    max_AUC_list = [0, 0, 0, 0]
    max_epoch_list = [0, 0, 0, 0]
    for epoch in range(1, NUM_EPOCHS + 1):
        print("epoch:"+str(epoch).zfill(2))
        model.load_weights(WRITE_SAVE_WEIGHT_DIR+str(epoch).zfill(2)+".hdf5")

        #混合行列出力
        #ROC出力
        AUC = [0, 0, 0, 0]
        for num_mbti, ydev in enumerate(ydev_list):
            print(str_ydev_list[num_mbti])
            print_confusion_matrix(xdev, ydev, num_mbti, model, MAX_SENTENCE_LENGTH)
            AUC[num_mbti] = print_roc_auc(xdev, ydev, num_mbti, model, MAX_SENTENCE_LENGTH)
             
            if max_AUC_list[num_mbti] < AUC[num_mbti]:
                max_AUC_list[num_mbti] = AUC[num_mbti]
                max_epoch_list[num_mbti] = epoch

    print("***BEST_EPOCH***")
    for num_mbti, ydev in enumerate(ydev_list):
        print(str_ydev_list[num_mbti])
        print("epoch\t\t= "+str(max_epoch_list[num_mbti]).zfill(2))
        print("AUC\t\t= "+str(max_AUC_list[num_mbti]))
        shutil.copyfile(WRITE_SAVE_WEIGHT_DIR+str(max_epoch_list[num_mbti]).zfill(2)+".hdf5", WRITE_WEIGHT_BEST_DIR+str_ydev_list[num_mbti]+"_"+str(max_epoch_list[num_mbti]).zfill(2)+".hdf5")
