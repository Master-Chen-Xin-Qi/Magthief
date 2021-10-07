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
from sklearn.svm import LinearSVC
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

    #y = keras.utils.to_categorical(y)
    svc = LinearSVC(random_state=0, tol=1e-5, verbose=1)
    svc.fit(X,y)

    predict_y_left = svc.predict(X_test_left)  # now do the final test
    #predict_y_left = oneHot2List(predict_y_left)
    predict_y_left = np.array(predict_y_left)

    y_test_left_formatrix = y_test_left[:,0]

    confusion = metrics.confusion_matrix(y_test_left_formatrix, predict_y_left)
    np.savetxt(model_folder + 'fcn_train_test_confusion_matrix.csv', confusion.astype(int), delimiter=',', fmt='%d')
    print('\tfinal confusion matrix:\n', confusion)
    precise,_,_,_,accuracy = validatePR(predict_y_left,y_test_left_formatrix)
    print('Accuracy :\n', accuracy)
    print('Precise :\n', precise)
    return


if __name__ == "__main__":
    train_fcn()
