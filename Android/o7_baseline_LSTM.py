# coding:utf-8
'''
@time:    Created on  2018-06-26 21:15:24
@author:  Lanqing
@Func:    使用简单模型实现复杂LSTM功能
                              和 FCN-LSTM使用相同数据源，一来节省时间，二来验证模型有效性，三来避免模型太复杂导致中间环节出问题。
          simple is best.
'''

import pickle
    
##### 加载参数，全局变量
with open('D:/Pycharm Projects/CODE/Android/config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    
    dict_all_parameters = pickle.load(f)

    train_tmp = dict_all_parameters['train_tmp']
    test_ratio = dict_all_parameters['test_ratio'] 
    test_ratio = dict_all_parameters['test_ratio'] 
    window_length = dict_all_parameters['window_length']
    epochs = dict_all_parameters['epochs'] 
    n_splits = dict_all_parameters['n_splits'] 
    model_folder = dict_all_parameters['model_folder'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    NB_CLASS = dict_all_parameters['NB_CLASS']
    units = dict_all_parameters['units'] 
    batch_size = dict_all_parameters['batch_size'] 
    train_batch_size = dict_all_parameters['train_batch_size'] 
    MAX_NB_VARIABLES = dict_all_parameters['MAX_NB_VARIABLES']

from CODE.Android.o7_baseline_traditional import validatePR
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import keras
import sklearn.utils.class_weight
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np
from collections import Counter

monitor = 'val_acc'
optimization_mode = 'max'
factor = 1. / np.sqrt(2)  # not time series 1. / np.sqrt(2)

def get_model():
    model = Sequential()
    model.add(LSTM(input_shape=(window_length, batch_size), units=units))
    model.add(Dropout(0.8))
    model.add(Dense(NB_CLASS, activation='softmax'))
    model.summary()
    optimize = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, decay=0.01)  # lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    # optimize = keras.optimizers.sgd()  # lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=optimize, metrics=['accuracy'])
    return model

def get_data():
    ###### 使用 prepare.py 处理好的、给FCN用的数据
    ###### 是没有OneHot的数据
    # data = np.load(train_tmp + 'data.npy')
    # label = np.load(train_tmp + 'label.npy')
    X_train = np.load(train_tmp + 'X_train.npy')
    y_train = np.load(train_tmp + 'y_train.npy')
    X_test = np.load(train_tmp + 'X_test.npy')
    y_test = np.load(train_tmp + 'y_test.npy') 
    return X_train[:, :window_length], y_train, X_test[:, :window_length], y_test

def get_full_dataset():
    ###### 使用 prepare.py 处理好的、给FCN用的数据
    ###### 是没有OneHot的数据
    # data = np.load(train_tmp + 'data.npy')
    # label = np.load(train_tmp + 'label.npy')
    X_train = np.load(train_tmp + 'X_train.npy')
    y_train = np.load(train_tmp + 'y_train.npy')
    X_test = np.load(train_tmp + 'X_test.npy')
    y_test = np.load(train_tmp + 'y_test.npy') 
    return X_train, y_train, X_test, y_test


def oneHot2List(labelOneHot):
    y_list = []
    for item in labelOneHot:
        y_list.append(np.argmax(item))
    return y_list

def get_weight(list_y):
    
    from sklearn.preprocessing import LabelEncoder

    classes = np.unique(list_y)
    le = LabelEncoder()
    y_ind = le.fit_transform(list_y)

    recip_freq = len(list_y) / (len(le.classes_) * 
                               np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    print("Class weights : ", class_weight)
    
    return class_weight

def train_lstm():
    
    X_train, y_train, X_test_left, y_test_left = get_data()
    X, y = X_train, y_train
    model = get_model()
    
    ######## 细粒度定义模型
    weight_fn = "%s/%s_lstm_weights.h5" % (model_folder, train_tmp.split('/')[-2])
    print(weight_fn)
    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    
    # y_oneHot = keras.utils.to_categorical(y)
    print('after one hot,y shape:', y.shape)
    
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    scores_accu, scores_f1 = [], []

    for i in range(2):
                
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        
        print('\n##########第%d次训练############\n' % i)
        print(dict(Counter(y_train[:, 0])))
        weight_dict = get_weight(list(y_train[:, 0]))
        
        y_train_new = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test_left)
        
        model.fit(X_train, y_train_new, batch_size=train_batch_size, epochs=20, verbose=1)  # , validation_data=(X_validate, y_validate))  # , class_weight=weight_dict)
        predict_y = model.predict(X_test_left) 
        predict_y = oneHot2List(predict_y)
        actual_y = oneHot2List(y_test)
        
        _, _, F1Score, _, accuracy_all = validatePR(predict_y, actual_y) 
        print(accuracy_all, F1Score)
        scores_accu.append(accuracy_all)
        scores_f1.append(scores_f1)

    print (' \n accuracy_all: \n', scores_accu, '\nMicro_average:  \n', scores_f1)  # judge model,get score
    
    #### 使用全部数据，使用保存的，模型进行实验
    predict_y_left = model.predict(X_test_left)  # now do the final test
    predict_y_left = oneHot2List(predict_y_left)
    s1 = metrics.accuracy_score(y_test_left, predict_y_left)
    f2 = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'lstm_train_test_confusion_matrix.csv', f2.astype(int), delimiter=',', fmt='%d')
    print ('accuracy_all:', s1, '\tfinal confusion matrix:\n', f2)

    return s1

if __name__ == '__main__':
    train_lstm()
