# coding:utf-8
'''
@time:    Created on  2018-06-30 14:35:08
@author:  Lanqing
@Func:    src.o7_hierarchical_fcn
'''

##### 加载参数，全局变量
import pickle
import tensorflow as tf
with open('D:/Pycharm Projects/CODE/Android/config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')

    dict_all_parameters = pickle.load(f)

    train_batch_size = dict_all_parameters['train_batch_size']
    MAX_NB_VARIABLES = dict_all_parameters['MAX_NB_VARIABLES']
    batch_size = dict_all_parameters['batch_size']
    train_tmp = dict_all_parameters['train_tmp']
    train_keyword = dict_all_parameters['train_keyword']
    train_tmp_test = dict_all_parameters['train_tmp_test']
    epochs = dict_all_parameters['epochs']
    window_length = dict_all_parameters['window_length']
    saved_dimension_after_pca = dict_all_parameters['saved_dimension_after_pca']

    n_splits = dict_all_parameters['n_splits']
    model_folder = dict_all_parameters['model_folder']
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test']
    evaluation_ratio = dict_all_parameters['evaluation_ratio']
    NB_CLASS = dict_all_parameters['NB_CLASS']

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings
from CODE.Android.Model import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, \
    min_max_scaler, one_hot_coding, PCA, train_test_evalation_split, knn_classifier, random_forest_classifier, \
    validatePR, check_model, generate_configs, vstack_list
from CODE.Android.main import read__data
from CODE.Android.config import predict_window, train_keyword, use_feature, train_folder, train_tmp, sigma, \
    overlap_window, window_length, model_folder, train_data_rate, train_folders, train_info_file, sample_rate, \
    batch_size, units, MAX_NB_VARIABLES, checkpoint_path, \
    saved_dimension_after_pca, NB_CLASS, whether_shuffle_train_and_test, M, time_step, APP_CLASS, evaluation_ratio, load_already_min_max_data,\
    one_label, use_time_and_fft, test_folder, train_or_test

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import  Lambda,Concatenate
from numpy.random import seed

from CODE.Android.o7_baseline_traditional import validatePR
from CODE.Android.o7_baseline_LSTM import get_full_dataset, get_weight, oneHot2List

import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import metrics
from keras.layers import MaxPool1D,Flatten
from keras import regularizers
seed(1)

##### FCN超参数

epochs = 50
# batch_size = batch_size
train_batch_size = 800
learning_rate = 0.5e-2
monitor = 'val_loss'  # acc
optimization_mode = 'min'
compile_model = True
factor = 1. / np.sqrt(2)  # not time series 1. / np.sqrt(2)

def expand(x):
    return tf.expand_dims(x, axis=-1)

def concat(l):
    inp = l[0]
    x_1 = l[1]
    y_tmp = l[2]
    label_in = y_tmp
    for i in range(1,time_step):
        inp_tmp = inp[:,:,i]
        inp_tmp = tf.expand_dims(inp_tmp, axis=-1)
        x_tmp, y_tmp = CNN(inp_tmp)
        x_tmp = tf.expand_dims(x_tmp, axis=-1)
        x_1x1 =  Conv1D(1,kernel_size=1, padding='same', kernel_initializer='he_uniform')(x_tmp)
        x_1 = concatenate([x_1, x_1x1])
        label_in = concatenate([label_in,y_tmp])
    return [x_1,label_in]

def CNN(inp):
    m= inp
    m = tf.expand_dims(m, axis=-1)
    y1 = Conv1D(16,kernel_size=5, padding='same', kernel_initializer='he_uniform')(m)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(32,kernel_size=3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16,kernel_size=3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    x1 = GlobalAveragePooling1D()(y1)
    y1 = Dense(NB_CLASS, activation='softmax')(x1)

    return [x1, y1]

def cnn_lstm():
    inp = Input(shape=(M,time_step))
    inapp_list = []
    vector_list = []
    for i in range(time_step):
        inp_tmp = Lambda(lambda x: x[:, :, i])(inp)
        l1 = Lambda(CNN)(inp_tmp)
        m = Lambda(expand)(l1[0])
        x_1 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(m)
        inapp_list.append(l1[1])
        vector_list.append(x_1)
    vector = Concatenate()(vector_list)
    label_inapp = Concatenate()(inapp_list)
    out_y = LSTM(16)(vector)
    label_app = Dense(APP_CLASS, activation='softmax')(out_y)
    model = Model(inputs=inp, outputs=[label_inapp,label_app])
    model.summary()
    return model

def generate_my_model_old():
    if use_time_and_fft:
        ip1 = Input(shape=(predict_window*window_length*2,1))
    else:
        ip1 = Input(shape=(predict_window*window_length,1))
    #y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(ip1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)
    y1 = MaxPool1D(2)(y1)

    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)
    y1 = MaxPool1D(2)(y1)

    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPool1D(2)(y1)

    y1 = Conv1D(8, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPool1D(2)(y1)
    y1 = Dropout(0.4)(y1)

    y1 = Flatten()(y1)

    out1 = Dense(32, activation='softmax')(y1)
    out = Dense(NB_CLASS, activation='softmax')(out1)

    model = Model(inputs=ip1, outputs=out)
    model.summary()
    return model

def generate_my_model():
    ip1 = Input(shape=(saved_dimension_after_pca,1))
    ip = Reshape((5,30))(ip1)
    out = LSTM(64, return_sequences=True)(ip)
    out = LSTM(32)(out)
    #out = Dense(128, activation='sigmoid')(out)
    out = Dense(NB_CLASS, activation='softmax')(out)
    # y1 = Flatten()(ip1)
    # out1 = Dense(512, activation='sigmoid')(y1)
    # out = Dense(NB_CLASS, activation='softmax')(out1)

    model = Model(inputs=ip1, outputs=out)
    model.summary()
    return model



def squeeze_excite_block(inputs):
    ''' Create a squeeze-excite block
    Args:
        inputs: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = inputs.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(inputs)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([inputs, se])
    return se


def result_saver(name, actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix):
    result_saver_dict = {}
    result_saver_dict['actual_y_list'] = actual_y_list
    result_saver_dict['prediction_y_list'] = prediction_y_list
    result_saver_dict['accuracy'] = accuracy
    result_saver_dict['loss'] = loss
    result_saver_dict['conf_matrix'] = conf_matrix
    print('conf_matrix：\n', conf_matrix)
    f = open(model_folder + name + '_final_result.txt', "w")
    for (key, value) in result_saver_dict.items():
        f.write(str(key) + '\t' + str(value) + '\n')
    f.close()
    return result_saver_dict


def train_evalation_split(data, label):
    '''
    split train and test
    :param data: train data
    :param label: train label
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_validate, y_train, y_validate = train_test_split(data, label, \
                                                                test_size=evaluation_ratio, random_state=0,
                                                                shuffle=True)
    return X_train, X_validate, y_train, y_validate


def train_fcn():
    if load_already_min_max_data == True:
    #载入已保存数据
        data = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/data.npy',allow_pickle=True)
        label = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/label.npy',allow_pickle=True)
        category_len = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len.npy',allow_pickle=True).item()
    else:
    #重新载入数据
        if train_or_test == 'train':
            data, label, category_len = read__data(train_folder, train_keyword, train_data_rate, train_tmp)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/data',data)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/label',label)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len',category_len)
        else:
            data, label, category_len = read__data(test_folder, train_keyword, train_data_rate, train_tmp)

    #一个标签
    if one_label == True:
        X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label, category_len)
        X, y = X_train, y_train
    else:
        in_app_label = label[:,0]
        app_label = label[:,1]
        X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label, category_len)
        X, y = X_train, y_train[:, 0]

    model = generate_my_model()
    #tmp_test = model.predict(X_test_left.reshape(len(X_test_left), saved_dimension_after_pca,1))
    #print(tmp_test)

    # y = keras.utils.to_categorical(y)
    # y_test_left = keras.utils.to_categorical(y_test_left)
    print('after one hot,y shape:', y.shape)

    n_splits = 10
    epochs = 500
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    scores_accu, scores_f1 = [], []
    ############# 十折交叉验证
    i = 0
    for_train = False #True时训练，False检验测试集
    if for_train == True:
        for train_index, test_index in skf_cv.split(X, y):

            i += 1
            print('============    第  %d 折交叉验证            =====================' % i)
            X_training, X_testing = X[train_index], X[test_index]
            y_training, y_test_orging = y[train_index], y[test_index]

            print(dict(Counter(y_training[:])))
            weight_dict = get_weight(list(y_training[:]))
            weight_fn = "%s/%s_fold%s_demo_5_30_64lstm_acc_weights.h5" % (model_folder, train_tmp.split('/')[-2],i)
            print(weight_fn)

            y_training = keras.utils.to_categorical(y_training)
            y_test_orging = keras.utils.to_categorical(y_test_orging)
            # X_train, X_validate, y_train, y_validate = train_evalation_split(X_train, y_train)
            print(X_training.shape)

            ######## 细粒度定义模型
            model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                               monitor=monitor, save_best_only=True, save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=15, mode=optimization_mode,
                                          factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
            callback_list = [model_checkpoint, reduce_lr]
            optm = Adam(lr=learning_rate)
            if compile_model:
                model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])

            #### 到这一步的时候，终于可以兄弟分家了
            if use_time_and_fft:
                X_training = X_training.reshape(len(X_training),2*predict_window*window_length,1)
                y_training = y_training
                X_val = X_testing.reshape(len(X_testing),2*predict_window*window_length,1)
                y_val = y_test_orging
            else:
                X_training= X_training.reshape(len(X_training),saved_dimension_after_pca,1)
                y_training = y_training
                X_val = X_testing.reshape(len(X_testing),saved_dimension_after_pca,1)
                y_val = y_test_orging
            model.fit(x=X_training[:len(X_training)], y=y_training[:len(X_training)], batch_size=256, epochs=epochs, callbacks=callback_list,
                     class_weight=weight_dict,
                      validation_data=[X_val[:len(X_val)],y_val[:len(X_val)]],
                      verbose=2)

    # X_test = X_test[:2090].reshape(-1,time_step,100)
    # predict_y = model.predict(X_train)
    # predict_y = oneHot2List(predict_y)
    # print(1)
    # precise = metrics.average_precision_score(y_test_org, predict_y)
    # report = metrics.classification_report()
    # _, _, F1Score, _, accuracy_all = validatePR(predict_y, y_test_org[0:2090:10])
    # print (' \n accuracy_all: \n', accuracy_all, '\F1Score:  \n', F1Score)  # judge model,get score
    # # print('report:', report)
    #
    # scores_accu.append(accuracy_all)
    # scores_f1.append(F1Score)

    # print (' \n accuracy_all: \n', scores_accu, '\nMicro_average:  \n', scores_f1)  # judge model,get score
    # ## 使用全部数据，使用保存的，模型进行实验
    model.load_weights('D:/Pycharm Projects/CODE/data/tmp/Android/model/train_fold10_demo_5_30_64lstm_acc_weights.h5')

    X_test_left = X_test_left.reshape(len(X_test_left), saved_dimension_after_pca,1)
    #X_test_left = X_train.reshape(len(X_train), saved_dimension_after_pca,1)

    predict_y_left = model.predict(X_test_left)  # now do the final test
    predict_y_left = oneHot2List(predict_y_left)
    predict_y_left = np.array(predict_y_left)

    y_test_left_formatrix = y_test_left
    #y_test_left_formatrix = y
    #y_test_left_formatrix = app_label

    confusion = metrics.confusion_matrix(y_test_left_formatrix, predict_y_left)
    np.savetxt(model_folder + 'fcn_train_test_confusion_matrix.csv', confusion.astype(int), delimiter=',', fmt='%d')
    print('\tfinal confusion matrix:\n', confusion)
    precise,_,_,_,accuracy = validatePR(predict_y_left,y_test_left_formatrix)
    print('Accuracy :\n', accuracy)
    print('Precise :\n', precise)
    return


if __name__ == "__main__":
    train_fcn()
