import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import seed

# from src.o7_baseline_traditional import validatePR
# from src.o7_baseline_LSTM import get_full_dataset, get_weight, oneHot2List

import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import metrics
from Model import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, min_max_scaler, one_hot_coding, PCA, train_test_evalation_split, knn_classifier, random_forest_classifier, validatePR, check_model, generate_configs, vstack_list
from main import read__data
from config import predict_window, train_keyword, use_feature, train_folder, train_tmp, sigma, overlap_window, window_length, model_folder, train_data_rate, train_folders, train_info_file, sample_rate, batch_size, units, MAX_NB_VARIABLES, checkpoint_path, \
saved_dimension_after_pca, NB_CLASS, whether_shuffle_train_and_test
def generate_model():
    ip1 = Input(shape=(window_length * predict_window, batch_size))
    ip2 = Input(shape=(saved_dimension_after_pca, batch_size))

    x1 = Masking()(ip1)
    x1 = LSTM(8)(x1)
    x1 = Dropout(0.8)(x1)

    y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(16, 8, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = GlobalAveragePooling1D()(y1)

    x1 = concatenate([x1, y1])

    #### 第二部分

    x2 = Masking()(ip2)
    x2 = LSTM(8)(x2)
    x2 = Dropout(0.8)(x2)

    y2 = Permute((2, 1))(ip2)
    y2 = Conv1D(16, 8, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = GlobalAveragePooling1D()(y2)
    x2 = concatenate([x2, y2])

    x = concatenate([x1, x2])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(inputs=[ip1, ip2], outputs=out)
    model.summary()

    # add load model code here to fine-tune

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

model = generate_model()