# coding:utf-8
'''
@time:    Created on  2021-04-05 17:12:34
@author:  Xinqi Chen
@Func:   CNN_LSTM model and train-val-test
'''

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Lambda, Input, Reshape
from tensorflow.python.keras.optimizers import Adam
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings
from CODE.Android.Model import  train_test_evalation_split, validatePR
from CODE.Android.main import read__data
from CODE.Android.config import predict_window, train_keyword, use_feature, train_folder, sigma, \
    overlap_window, window_length, model_folder, train_data_rate, \
    saved_dimension_after_pca, NB_CLASS, whether_shuffle_train_and_test, M, time_step, APP_CLASS, evaluation_ratio, load_already_min_max_data,\
    one_label, use_time_and_fft, test_folder, train_or_test

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Concatenate, Lambda
from CODE.Android.o7_baseline_traditional import validatePR
from CODE.Android.o7_baseline_LSTM import get_full_dataset, get_weight, oneHot2List

import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import metrics
from keras.callbacks import  EarlyStopping

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

def expand(x):
    return tf.expand_dims(x, axis=-1)

def reduce(x):
    return tf.squeeze(x, axis=-1)

def transpose(x):
    return tf.transpose(x,[0,2,1])

def triplet_loss(y_true, y_pred):
    y_pred = keras.backend.l2_normalize(y_pred,axis=1)
    batch = 256
    #print(batch)
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = keras.backend.sum(keras.backend.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = keras.backend.sum(keras.backend.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = keras.backend.sqrt(dis_pos)
    dis_neg = keras.backend.sqrt(dis_neg)
    a1 = 17
    d1 = dis_pos + keras.backend.maximum(0.0, dis_pos - dis_neg + a1)
    return keras.backend.mean(d1)

def CNN(inp):
    m = inp
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

    y1 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    x1 = GlobalAveragePooling1D()(y1)
    x1 = Dense(256, activation='softmax')(x1)
    y1 = Dense(NB_CLASS, activation='softmax')(x1)


    return [x1, y1]

def cnn_lstm_old():
    inp = Input(shape=(time_step,M))
    inapp_list = []
    vector_list = []

    # 第一个CNN
    inp_tmp1 = Lambda(lambda x: x[:, 0, :])(inp)
    #l1 = Lambda(CNN)(inp_tmp) CNN返回两个tensor，l1[0]为要传给lstm的向量，l1[1]为inapp标签
    n1 = Lambda(expand)(inp_tmp1)
    n1 = Permute((2, 1))(n1)
    y1 = Conv1D(16, kernel_size=5, padding='same', kernel_initializer='he_uniform')(n1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    x1 = GlobalAveragePooling1D()(y1)
    x1 = Dense(128, activation='softmax')(x1)
    out1 = Dense(NB_CLASS, activation='softmax')(x1)
    m1 = Lambda(expand)(x1)

    # 第二个CNN
    inp_tmp2 = Lambda(lambda x: x[:, 1, :])(inp)
    n2 = Lambda(expand)(inp_tmp2)
    n2 = Permute((2, 1))(n2)
    y2 = Conv1D(16, kernel_size=5, padding='same', kernel_initializer='he_uniform')(n2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    x2 = GlobalAveragePooling1D()(y2)
    x2 = Dense(128, activation='softmax')(x2)
    out2 = Dense(NB_CLASS, activation='softmax')(x2)
    m2 = Lambda(expand)(x2)

    # 第三个CNN
    inp_tmp3 = Lambda(lambda x: x[:, 2, :])(inp)
    n3 = Lambda(expand)(inp_tmp3)
    n3 = Permute((2, 1))(n3)
    y3 = Conv1D(16, kernel_size=5, padding='same', kernel_initializer='he_uniform')(n3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = squeeze_excite_block(y3)

    y3 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = squeeze_excite_block(y3)

    y3 = Conv1D(16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = squeeze_excite_block(y3)

    y3 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)

    x3 = GlobalAveragePooling1D()(y3)
    x3 = Dense(128, activation='softmax')(x3)
    out3 = Dense(NB_CLASS, activation='softmax')(x3)
    m3 = Lambda(expand)(x3)

    # 第四个CNN
    inp_tmp4 = Lambda(lambda x: x[:, 3, :])(inp)
    n4 = Lambda(expand)(inp_tmp4)
    n4 = Permute((2, 1))(n4)
    y4 = Conv1D(16, kernel_size=5, padding='same', kernel_initializer='he_uniform')(n4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = squeeze_excite_block(y4)

    y4 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = squeeze_excite_block(y4)

    y4 = Conv1D(16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = squeeze_excite_block(y4)

    y4 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)

    x4 = GlobalAveragePooling1D()(y4)
    x4 = Dense(128, activation='softmax')(x4)
    out4 = Dense(NB_CLASS, activation='softmax')(x4)
    m4 = Lambda(expand)(x4)

    inapp_list.append(out1)
    inapp_list.append(out2)
    inapp_list.append(out3)
    inapp_list.append(out4)
    vector_list.append(m1)
    vector_list.append(m2)
    vector_list.append(m3)
    vector_list.append(m4)
    #合并所有time step的向量
    vector = Concatenate()(vector_list)
    #把每个tensor转为层输出，reduce减少最后的维度
    label_inapp1 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[0]))
    label_inapp1 = Lambda(reduce)(label_inapp1)
    label_inapp2 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[1]))
    label_inapp2 = Lambda(reduce)(label_inapp2)
    label_inapp3 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[2]))
    label_inapp3 = Lambda(reduce)(label_inapp3)
    label_inapp4 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[3]))
    label_inapp4 = Lambda(reduce)(label_inapp4)
    #合并四个输出层
    # label_inapp_total = Concatenate([label_inapp1,label_inapp2])
    # label_inapp_total = Concatenate([label_inapp_total, label_inapp3])
    # label_inapp_total = Concatenate([label_inapp_total, label_inapp4])

    #将(None,units,time step)的数据模式变为(None,time step,units)，因为lstm接收第二维是time step
    # vector = Lambda(transpose)(vector)

    out_y = LSTM(32)(vector) #16为神经元个数
    out_y = Dense(128, activation='softmax')(out_y)
    label_app = Dense(APP_CLASS, activation='softmax')(out_y)
    model = Model(inputs=inp, outputs=[label_inapp1,label_inapp2,label_inapp3,label_inapp4,label_app])

    model.summary()
    return model

def cnn_lstm():
    inapp1 = Input(shape=(saved_dimension_after_pca,1),name='in_inapp1')
    inapp2 = Input(shape=(saved_dimension_after_pca, 1),name='in_inapp2')
    inapp3 = Input(shape=(saved_dimension_after_pca, 1),name='in_inapp3')
    inapp4 = Input(shape=(saved_dimension_after_pca, 1),name='in_inapp4')
    inapp_list = []
    vector_list = []

    # 第一个LSTM
    inp_tmp1 = inapp1
    #l1 = Lambda(CNN)(inp_tmp) CNN返回两个tensor，l1[0]为要传给lstm的向量，l1[1]为inapp标签
    n1 = Reshape((5,30))(inp_tmp1)
    y1 = LSTM(64, return_sequences=True)(n1)
    y1 = LSTM(32)(y1)
    out1 = Dense(NB_CLASS, activation='softmax',name='out_inapp1')(y1)
    m1 = Lambda(expand)(y1)

    # 第二个CNN
    #inp_tmp2 = Lambda(lambda x: x[:, 1, :])(inp)
    inp_tmp2 = inapp2
    n2 = Reshape((5,30))(inp_tmp2)
    y2 = LSTM(64, return_sequences=True)(n2)
    y2 = LSTM(32)(y2)
    out2 = Dense(NB_CLASS, activation='softmax',name='out_inapp2')(y2)
    m2 = Lambda(expand)(y2)

    # 第三个CNN
    inp_tmp3 = inapp3
    n3 = Reshape((5, 30))(inp_tmp3)
    y3 = LSTM(64, return_sequences=True)(n3)
    y3 = LSTM(32)(y3)
    out3 = Dense(NB_CLASS, activation='softmax',name='out_inapp3')(y3)
    m3 = Lambda(expand)(y3)

    # 第四个CNN
    inp_tmp4 = inapp4
    n4 = Reshape((5, 30))(inp_tmp4)
    y4 = LSTM(64, return_sequences=True)(n4)
    y4 = LSTM(32)(y4)
    out4 = Dense(NB_CLASS, activation='softmax',name='out_inapp4')(y4)
    m4 = Lambda(expand)(y4)

    inapp_list.append(out1)
    inapp_list.append(out2)
    inapp_list.append(out3)
    inapp_list.append(out4)
    #合并所有time step的向量
    vector = keras.layers.concatenate([m1,m2,m3,m4])
    vector = Permute((2,1))(vector)
    #把每个tensor转为层输出，reduce减少最后的维度
    # label_inapp1 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[0]))
    # label_inapp1 = Lambda(reduce, name='out_inapp1')(label_inapp1)
    # label_inapp2 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[1]))
    # label_inapp2 = Lambda(reduce, name='out_inapp2')(label_inapp2)
    # label_inapp3 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[2]))
    # label_inapp3 = Lambda(reduce, name='out_inapp3')(label_inapp3)
    # label_inapp4 = Conv1D(1, kernel_size=1, padding='same', kernel_initializer='he_uniform')(Lambda(expand)(inapp_list[3]))
    # label_inapp4 = Lambda(reduce, name='out_inapp4')(label_inapp4)
    #合并四个输出层
    # label_inapp_total = Concatenate([label_inapp1,label_inapp2])
    # label_inapp_total = Concatenate([label_inapp_total, label_inapp3])
    # label_inapp_total = Concatenate([label_inapp_total, label_inapp4])

    #将(None,units,time step)的数据模式变为(None,time step,units)，因为lstm接收第二维是time step
    # vector = Lambda(transpose)(vector)

    out_y = LSTM(32)(vector) #16为神经元个数
    out_y = Dense(128, activation='softmax')(out_y)
    label_app = Dense(APP_CLASS, activation='softmax', name='out_app')(out_y)
    model = Model(inputs=[inapp1,inapp2,inapp3,inapp4], outputs=[out1,out2,out3,out4,label_app])

    model.summary()
    return model

learning_rate = 1e-2 #学习率
monitor = 'val_loss'  # acc
optimization_mode = 'min'
compile_model = True
factor = 1. / np.sqrt(2)  # not time series 1. / np.sqrt(2)
def train_fcn():
    train_tmp = '../data//tmp/Android//tmp/train/'
    if load_already_min_max_data == True:
    #载入已保存数据
        data = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/data.npy', allow_pickle=True)
        label = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/label.npy', allow_pickle=True)
        category_len = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len.npy', allow_pickle=True).item()
    else:
    #重新载入数据
        if train_or_test == 'train':
            data, label, category_len = read__data(train_folder, train_keyword, train_data_rate, train_tmp)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/data.npy',data)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/label.npy',label)
            np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len.npy',category_len)
        else:
            data, label, category_len = read__data(test_folder, train_keyword, train_data_rate, train_tmp)

    #一个标签
    if one_label == True:
        X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label, category_len)
        X, y = X_train, y_train
    else:
        X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label, category_len)
        X, y = X_train, y_train[:, 0]
        app_label = y_train[:, 1]
    model = cnn_lstm()
    keras.utils.plot_model(model, "cnn_lstm_model.png", show_shapes=True)

    # 检验模型中间层输出
    # intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('lambda_4').output)
    # data = intermediate_layer_model.predict(X_train[:4].reshape(1,4,M))
    # print(data)

    # 检验模型输出
    #tmp_test = model.predict(X_test_left.reshape(-1,M,time_step))
    #print(tmp_test)

    print('after one hot,y shape:', y.shape)

    n_splits = 10
    epochs = 500
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    ############ 十折交叉验证 ############
    i = 0
    for_train = False #True时训练，False检验测试集
    if for_train == True:
        for train_index, test_index in skf_cv.split(X, y):
            i += 1
            print('============    第  %d 折交叉验证    =====================' % i)
            X_training, X_testing = X[train_index], X[test_index]
            y_training_inapp, y_test_orging_inapp = y[train_index], y[test_index]
            y_training_app, y_test_orging_app = app_label[train_index], app_label[test_index]

            print(dict(Counter(y_training_inapp[:])))
            weight_dict_inapp = get_weight(list(y_training_inapp[:]))
            weight_dict_app = get_weight(list(y_training_app[:]))
            weight_fn = "%s/%s_new_cnn_lstm_weights.h5" % (model_folder, train_tmp.split('/')[-2])
            print(weight_fn)
            # X_train, X_validate, y_train, y_validate = train_evalation_split(X_train, y_train)
            print(X_training.shape)

            ######## 细粒度定义模型
            model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                               monitor=monitor, save_best_only=True, save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=15, mode=optimization_mode,
                                          factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
            early_stopping = EarlyStopping(monitor='val_loss', patience=30)
            callback_list = [model_checkpoint, reduce_lr, early_stopping]
            optm = Adam(lr=learning_rate)
            if compile_model:
                model.compile(optimizer=optm, loss={'out_inapp1':'categorical_crossentropy',
                                                    'out_inapp2':'categorical_crossentropy',
                                                    'out_inapp3': 'categorical_crossentropy',
                                                    'out_inapp4': 'categorical_crossentropy',
                                                    'out_app': 'categorical_crossentropy'},
                              metrics=[keras.metrics.CategoricalAccuracy()])

            #转化成one-hot编码
            y_training_inapp = y_training_inapp[:len(y_training_inapp)//time_step*time_step]
            y_training_app = y_training_app[:len(y_training_app)//time_step*time_step]
            y_training_inapp_label = keras.utils.to_categorical(y_training_inapp)
            y_training_app_label = keras.utils.to_categorical(y_training_app)

            # y_training_inapp_label1 = y_training_inapp_label[:, 0, :].reshape(-1, NB_CLASS)
            # y_training_inapp_label2 = y_training_inapp_label[:, 1, :].reshape(-1, NB_CLASS)
            # y_training_inapp_label3 = y_training_inapp_label[:, 2, :].reshape(-1, NB_CLASS)
            # y_training_inapp_label4 = y_training_inapp_label[:, 3, :].reshape(-1, NB_CLASS)
            # y_training_app_label = y_training_app_label[:, 0, :].reshape(-1, APP_CLASS)

            y_test_orging_inapp = y_test_orging_inapp[:(len(y_test_orging_inapp)//time_step)*time_step]
            y_test_orging_app = y_test_orging_app[:(len(y_test_orging_app)//time_step)*time_step]
            y_val_inapp_label = keras.utils.to_categorical(y_test_orging_inapp)
            y_val_app_label = keras.utils.to_categorical(y_test_orging_app)

            # y_val_inapp_label1 = y_val_inapp_label[:,0,:].reshape(-1,NB_CLASS)
            # y_val_inapp_label2 = y_val_inapp_label[:, 1, :].reshape(-1, NB_CLASS)
            # y_val_inapp_label3 = y_val_inapp_label[:, 2, :].reshape(-1, NB_CLASS)
            # y_val_inapp_label4 = y_val_inapp_label[:, 3, :].reshape(-1, NB_CLASS)
            # y_val_app_label = y_val_app_label[:, 0, :].reshape(-1, APP_CLASS)

            # 每一段数据切分成time step个in_app标签
            # y_training_inapp_label_timestep = []
            # y_val_inapp_label_timestep = []
            # for i in range(len(y_training_inapp_label)):
            #     train_tmp = []
            #     for j in range(time_step):
            #         train_tmp.append(y_training_inapp_label[i])
            #     y_training_inapp_label_timestep.append(train_tmp)
            # for i in range(len(y_val_inapp_label)):
            #     val_tmp = []
            #     for j in range(time_step):
            #         val_tmp.append(y_val_inapp_label[i])
            #     y_val_inapp_label_timestep.append(val_tmp)

            # y_training_inapp_label_timestep = np.array(y_training_inapp_label_timestep)
            # y_val_inapp_label_timestep = np.array(y_val_inapp_label_timestep)

            #### 到这一步的时候，终于可以兄弟分家了
            if use_time_and_fft:
                X_training = X_training.reshape(len(X_training)//time_step*time_step,time_step,2*M*time_step)
                X_val = X_testing.reshape(len(X_testing)//time_step,time_step,2*M*time_step)
            else:
                #训练集
                X_len = len(X_training)//time_step*time_step
                X_training= X_training[:X_len]
                in_app1 = X_training#[0:-1:4]
                in_app1 = in_app1.reshape(len(in_app1),150,1)
                in_app2 = X_training#[1:-1:4]
                in_app2 = in_app2.reshape(len(in_app1), 150, 1)
                in_app3 = X_training#[2:-1:4]
                in_app3 = in_app3.reshape(len(in_app1), 150, 1)
                in_app4 = X_training#[3:len(X_training):4]
                in_app4 = in_app4.reshape(len(in_app1), 150, 1)
                #验证集
                X_val_len = len(X_testing)//time_step*time_step
                X_val = X_testing[:X_val_len]
                val_in_app1 = X_val#[0:-1:4]
                val_in_app1 = val_in_app1.reshape(len(val_in_app1), 150,1)
                val_in_app2 = X_val#[1:-1:4]
                val_in_app2 = val_in_app2.reshape(len(val_in_app1), 150, 1)
                val_in_app3 = X_val#[2:-1:4]
                val_in_app3 = val_in_app3.reshape(len(val_in_app1), 150, 1)
                val_in_app4 = X_val#[3:len(X_training):4]
                val_in_app4 = val_in_app4.reshape(len(val_in_app1), 150, 1)
            model.fit({"in_inapp1":in_app1,
                       "in_inapp2":in_app2,
                       "in_inapp3":in_app3,
                       "in_inapp4":in_app4},
                        #数据不同则加上[0:-1:4]  [3:len(y_training_inapp_label):4]等等
                      {"out_inapp1":y_training_inapp_label.reshape(len(in_app1),NB_CLASS),
                       "out_inapp2": y_training_inapp_label.reshape(len(in_app1),NB_CLASS),
                        "out_inapp3": y_training_inapp_label.reshape(len(in_app1),NB_CLASS),
                        "out_inapp4": y_training_inapp_label.reshape(len(in_app1),NB_CLASS),
                        # 每次四段数据都是取自同一app
                        "out_app": y_training_app_label.reshape(len(in_app1),APP_CLASS)},
                                       batch_size=256, epochs=epochs, callbacks=callback_list,
                      class_weight=[1,1,1,1,2],#最终的app weight最大
                      validation_data=[[val_in_app1,val_in_app2,val_in_app3,val_in_app4],
                                       [y_val_inapp_label.reshape(len(val_in_app1),NB_CLASS),
                                              y_val_inapp_label.reshape(len(val_in_app1),NB_CLASS),
                                              y_val_inapp_label.reshape(len(val_in_app1),NB_CLASS),
                                              y_val_inapp_label.reshape(len(val_in_app1),NB_CLASS),
                                              y_val_app_label.reshape(len(val_in_app1),APP_CLASS)]],
                      verbose=2)

    # 使用全部数据，使用保存的，模型进行实验
    model.load_weights('D:/Pycharm Projects/CODE/data/tmp/Android/model/train_cnn_lstm_weights.h5')
    X_test_len = len(X_test_left)//time_step*time_step
    X_test_left = X_test_left[:X_test_len]
    X_test_left = X_test_left.reshape(len(X_test_left),saved_dimension_after_pca,1)
    # 预测
    predict_y_left = model.predict([X_test_left,X_test_left,X_test_left,X_test_left])  # now do the final test
    # 提取每个time step的标签
    predict_y_left_inapp1 = np.array(oneHot2List(predict_y_left[0]))
    predict_y_left_inapp2 = np.array(oneHot2List(predict_y_left[1]))
    predict_y_left_inapp3 = np.array(oneHot2List(predict_y_left[2]))
    predict_y_left_inapp4 = np.array(oneHot2List(predict_y_left[3]))
    # 合并所有time step的inapp标签
    predict_y_left_inapp = np.hstack((predict_y_left_inapp1,predict_y_left_inapp2))
    predict_y_left_inapp = np.hstack((predict_y_left_inapp, predict_y_left_inapp3))
    predict_y_left_inapp = np.hstack((predict_y_left_inapp, predict_y_left_inapp4))
    predict_y_left_app = np.array(oneHot2List(predict_y_left[4]))

    y_test_left_inapp = y_test_left[:,0]
    y_test_left_app_formatrix = y_test_left[:,1]
    # inapp混淆矩阵与准确率
    y_test_left_inapp_formatrix = np.array(y_test_left_inapp)
    y_test_left_inapp_formatrix = np.hstack((y_test_left_inapp_formatrix,y_test_left_inapp_formatrix))
    y_test_left_inapp_formatrix = np.hstack((y_test_left_inapp_formatrix,y_test_left_inapp_formatrix))
    confusion_inapp = metrics.confusion_matrix(predict_y_left_inapp, y_test_left_inapp_formatrix)
    np.savetxt(model_folder + 'cnn_lstm_test_inapp_confusion_matrix.csv', confusion_inapp.astype(int), delimiter=',', fmt='%d')
    print('\tfinal inapp confusion matrix:\n', confusion_inapp)
    precise,_,_,_,accuracy = validatePR(predict_y_left_inapp, y_test_left_inapp_formatrix)
    print('Inapp Accuracy :\n', accuracy)
    print('Inapp Precise :\n', precise)
    # app混淆矩阵与准确率
    # a = len(y_test_left_app_formatrix) // time_step * time_step
    y_test = y_test_left_app_formatrix
    confusion_app = metrics.confusion_matrix(predict_y_left_app, y_test)
    np.savetxt(model_folder + 'cnn_lstm_test_app_confusion_matrix.csv', confusion_app.astype(int), delimiter=',', fmt='%d')
    print('\tfinal inapp confusion matrix:\n', confusion_app)
    precise,_,_,_,accuracy = validatePR(predict_y_left_app, y_test)
    print('App Accuracy :\n', accuracy)
    print('App Precise :\n', precise)
    return

if __name__ == "__main__":
    train_fcn()