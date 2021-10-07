# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    Read data and Preprocess
'''
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np
from sklearn import metrics
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from mlp.mlp_solver import Solver
import sys
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import seed
from scipy import signal

from CODE.Android.config import predict_window, train_keyword, use_feature, train_folder, train_tmp, sigma, overlap_window, \
    window_length, model_folder, train_data_rate, train_folders, train_info_file, sample_rate, batch_size, units, \
    MAX_NB_VARIABLES, checkpoint_path, time_step, saved_dimension_after_pca, \
    NB_CLASS, M, APP_CLASS, load_already_min_max_data, app_keyword, one_label, use_pca, use_time_and_fft
from CODE.Android.Model import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, min_max_scaler, \
    one_hot_coding, PCA, train_test_evalation_split, knn_classifier, random_forest_classifier, validatePR, check_model, \
    generate_configs, vstack_list


def CNN(inp):
    y1 = Conv1D(8, kernel_size=(1, 3), padding='same', kernel_initializer='he_uniform')(inp)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, kernel_size=(1, 3), padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, kernel_size=(1, 3), padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    x1 = GlobalAveragePooling1D()(y1)
    y1 = Dense(NB_CLASS, activation='softmax')(y1)
    return x1, y1


def cnn_lstm(inp):
    x = []
    x_1 = None
    label_in = []
    for i in range(time_step):
        x_tmp, y_tmp = CNN(inp[:, :, i])
        y_tmp = Dense(NB_CLASS, activation='softmax')(y_tmp)
        x_1x1 = Conv1D(16, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform')(x_tmp)
        x.append(x_tmp)
        x_1 = concatenate(x_1, x_1x1)
        label_in.append(y_tmp)
    out_y = LSTM(time_step)(x_1)
    label_app = Dense(APP_CLASS, activation='softmax')(out_y)
    return label_in, label_app


def generate_my_model():
    ip1 = Input(shape=(time_step, saved_dimension_after_pca // time_step))
    # ip2 = Input(shape=(saved_dimension_after_pca, batch_size))

    x1 = Masking()(ip1)
    x1 = LSTM(8)(x1)
    x1 = Dropout(0.5)(x1)

    y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(32, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = Conv1D(8, 1, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = GlobalAveragePooling1D()(y1)

    x1 = concatenate([x1, y1])

    out = Dense(NB_CLASS, activation='softmax')(x1)

    model = Model(inputs=ip1, outputs=out)
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

def app_name_process(file_name):
    for m in app_keyword:
        if m in file_name:
            return m

def single_file_process(array_, acc_array, category, after_fft_data):
    '''
        1. Process after "vstack"
        2. receive an file array and corresponding category
        3. return a clean array
    '''
    final_list = []
    numerical_feature_list = []
    rows, cols = array_.shape
    i = 0
    while (i * overlap_window + window_length < rows):  # split window, attention here
        tmp_window = array_[(i * overlap_window): (i * overlap_window + window_length)]
        tmp_window = tmp_window.reshape([1, window_length * cols])

        #acc_window = acc_array[(i * overlap_window): (i * overlap_window + window_length), :]
        ###################
        # 消除地磁
        # tmp_window = tmp_window - Solver().predict(acc_window, checkpoint_path)
        ###################
        #tmp_window = tmp_window-np.mean(tmp_window)
        # gauss filter
        tmp_window = gauss_filter(tmp_window, sigma)
        numerical_feature_tmp = tmp_window
        # fft process
        fft_window = fft_transform(tmp_window)
        #tmp_window = tmp_window[1:]
        final_list.append(fft_window)
        numerical_feature_list.append(numerical_feature_tmp.T)
        i += 1
    final_array = np.array(final_list)
    numerical_feature_list = np.array(numerical_feature_list)
    final_array = final_array.reshape([final_array.shape[0], final_array.shape[1]])
    numerical_feature_array = numerical_feature_list.reshape(
        [numerical_feature_list.shape[0], numerical_feature_list.shape[1]])
    return final_array, numerical_feature_array


def read__data(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
        4. 完成修改，默认读入一个一维的矩阵，降低采样率
    '''
    file_dict = divide_files_by_name(input_folder, different_category)
    cols = predict_window
    fft_list, num_list, label = [], [], []
    category_len = {}  # 保存各个类别的数据长度，为了方便划分训练集和测试集
    if one_label == False:
        for category in different_category:
            for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
                app_label = app_name_process(one_category_single_file) #app标签
                file_array, acc_array = read_single_txt_file_new(one_category_single_file)
                fft_feature, num_feature = single_file_process(file_array, acc_array, category,
                                                               after_fft_data_folder)  # 预处理
                tmp_label = [(category, app_label)] * len(fft_feature)
                name = category+'_'+app_label
                fft_list.append(fft_feature)
                num_list.append(num_feature)
                label += tmp_label
                length = len(fft_feature)
                if name not in category_len:
                    category_len[name] = 0
                category_len[name] += length

        fft_data = vstack_list(fft_list)
        data_feature = vstack_list(num_list)

        if use_pca == True:
            fft_data = PCA(fft_data)
            data_feature = PCA(data_feature)
        ''' Attention Here, Using FFT Only '''
        ''' Changes Here '''
        if use_feature == 'FFT':
            data = fft_data
        else:
            print(data_feature.shape, fft_data.shape)
            data = np.hstack((data_feature, fft_data))

        label = np.array(label)
        inapp_label = list(label[:, 0])
        app_type_label = list(label[:, 1])
        #### 暂时苟合在一起， 最后处理、划分完训练、预测集再分开； 先FFT，后numberic
        enc1 = LabelEncoder()
        enc2 = LabelEncoder()
        inapp_label = enc1.fit_transform(inapp_label).reshape(len(inapp_label), 1)
        app_type_label = enc2.fit_transform(app_type_label).reshape(len(app_type_label), 1)
        total_label = np.hstack((inapp_label, app_type_label))
        # inapp_label, _ = one_hot_coding(inapp_label, 'train')  # not really 'one-hot',haha
        # app_type_label, _ = one_hot_coding(app_type_label, 'not_train')
        print('Shape of data,shape of label:', data.shape, total_label.shape)
    else:
        for category in different_category:
            file_array_one_category = np.array([[0]] * cols).T  # Initial, skill here
            acc_array_one_category = np.array([[0]] * 3).T
            for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
                file_array, acc_array = read_single_txt_file_new(one_category_single_file)
                file_array_one_category = np.vstack((file_array_one_category, file_array))
                #acc_array_one_category = np.vstack((acc_array_one_category, acc_array))
            file_array_one_category = file_array_one_category[1:]  # exclude first line
            #acc_array_one_category = acc_array_one_category[1:]
            fft_feature, num_feature = single_file_process(file_array_one_category, acc_array_one_category, category,
                                                         after_fft_data_folder)  # 预处理
            #print(fft_feature.shape)
            tmp_label = [category] * len(fft_feature)
            category_len[category] = len(fft_feature)
            # generate label part and merge all
            fft_list.append(fft_feature)
            num_list.append(num_feature)
            label += tmp_label

        fft_data = vstack_list(fft_list)
        data_feature = vstack_list(num_list)

        if use_pca == True:
            fft_data = PCA(fft_data)
            data_feature = PCA(data_feature)
        ''' Attention Here, Using FFT Only '''
        ''' Changes Here '''
        if use_feature == 'FFT':
            data = fft_data
        else:
            print(data_feature.shape, fft_data.shape)
            data = np.hstack((data_feature, fft_data))
        label = np.array(label)
        #### 暂时苟合在一起， 最后处理、划分完训练、预测集再分开； 先FFT，后numberic
        enc1 = LabelEncoder()
        label = enc1.fit_transform(label)
        total_label, _ = one_hot_coding(label, 'train')  # not really 'one-hot',haha
        print('Shape of data,shape of label:', data.shape, label.shape)
    fft_data_total = min_max_scaler(fft_data)
    time_data_total = min_max_scaler(data_feature)
    if use_time_and_fft:
        fft_time_data = np.hstack((fft_data, np.abs(data_feature)))  # 时频信号整合
        fft_time_data = min_max_scaler(fft_time_data)
    else:
        fft_time_data = fft_data_total
    #每一行Min_Max
    # fft_data_total = np.array([0] * fft_data.shape[1]).reshape(1, saved_dimension_after_pca)
    # for i in range(len(data)):
    #     data_tmp = fft_data[i, :].T
    #     data_tmp = data_tmp.reshape(saved_dimension_after_pca, 1)
    #     r, c = data_tmp.shape
    #     train_data = data_tmp.reshape([r * c, 1])
    #     XX = preprocessing.MinMaxScaler().fit(train_data)
    #     train_data = XX.transform(train_data)
    #     data_tmp = train_data.reshape([r, c])
    #     # data_tmp = min_max_scaler(data_tmp)
    #     data_total = np.vstack((fft_data_total, data_tmp.T))
    #     if (i % 10000 == 0):
    #         print("%d已完成" % i)
    # print('Min Max Finished!')
    # fft_data_total = fft_data_total[1:]

    #return fft_data_total, label, category_len
    return fft_time_data, total_label, category_len

def baseline_trainTest(data, label, category_len):
    """
        Train and evaluate using KNN and RF classifier
    """
    X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label, category_len)
    X = X_train
    y = y_train
    print('All samples shape: ', data.shape)
    file_write = model_folder + 'best_model'
    model_save_file = file_write
    model_save = {}
    train_wt = None
    num_train = X.shape[0]
    is_binary_class = (len(np.unique(y)) == 2)
    # test_classifiers = ['NB','KNN', 'LR', 'RF', 'DT','SVM','GBDT','AdaBoost']
    test_classifiers = ['RF']  # , 'DT','LR']  # , 'GBDT', 'AdaBoost']
    classifiers = {'KNN': knn_classifier,
                   'RF': random_forest_classifier}
    # print ('******************** Data Info *********************')
    scores_Save = []
    model_dict = {}
    accuracy_all_list = []
    for classifier in test_classifiers:
        # print ('******************* %s ********************' % classifier)
        scores = []
        skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        # print(skf_cv)
        if(one_label == False):
            y = y[:, 0]
        i = 0
        for train_index, test_index in skf_cv.split(X, y):
            i += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = classifiers[classifier](X_train, y_train, train_wt)
            predict_y = model.predict(X_test)
            Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y, y_test)
            # print(accuracy_all, F1Score)
            # print (' \n Precise: %f \n' % Precise, 'Recall: %f \n' % Recall, 'F1Score: %f \n' % F1Score)  # judge model,get score
            scores.append({'cnt:': i, 'mean-F1-Score': Micro_average, 'accuracy-all': accuracy_all})
            accuracy_all_list.append(accuracy_all)
            print("第%d次训练完毕" % i)
        Micro_average, accuracyScore = [], []
        for item in scores:
            Micro_average.append(item['mean-F1-Score'])
            accuracyScore.append(item['accuracy-all'])
        Micro_average = np.mean(Micro_average)
        accuracyScore = np.mean(accuracyScore)
        scoresTmp = [accuracy_all, Micro_average]
        # print (' \n accuracy_all: \n', accuracy_all, '\nMicro_average:  \n', Micro_average)  # judge model,get score
        scores_Save.append(scoresTmp)
        model_dict[classifier] = model
    print('******************* End ********************')
    scores_Save = np.array(scores_Save)
    max_score = np.max(scores_Save[:, 1])
    index = np.where(scores_Save == np.max(scores_Save[:, 1]))
    index_model = index[0][0]
    model_name = test_classifiers[index_model]
    # print (' \n Best model: %s \n' % model_name)
    print('Test accuracy: ', max_score)
    joblib.dump(model_dict[model_name], file_write)
    ######## 重新调整，打印混淆矩阵
    model_sort = []
    scores_Save1 = scores_Save * (-1)
    sort_Score1 = np.sort(scores_Save1[:, 1])  # inverse order
    for item in sort_Score1:
        index = np.where(scores_Save1 == item)
        index = index[0][0]
        model_sort.append(test_classifiers[index])
    #### 使用全部数据，使用保存的，模型进行实验
    model = model_dict[model_name]
    predict_y_left = model.predict(X_test_left)  # now do the final test
    # Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y_left, y_test_left)
    # print ('\n final test: model: %s, F1-mean: %f,accuracy: %f' % (model_sort[0], Micro_average, accuracy_all))
    s1 = metrics.accuracy_score(y_test_left, predict_y_left)
    f2 = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'traditional_train_test_confusion_matrix.csv', f2.astype(int), delimiter=',', fmt='%d')
    # f1 = metrics.fbeta_score(y_test_left, predict_y_left,beta= 0.5)
    # print ('Not mine: final test: model: %s,\n accuracy: %f' % (model_sort[0], s1),)
    print('Matrix:\n', f2.astype(int))
    return accuracy_all_list, max_score


def main():
    '''
        Complete code for processing one folder
    '''
    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    if load_already_min_max_data == True:
        data = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/data.npy', allow_pickle=True)
        label = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/label.npy', allow_pickle=True)
        category_len = np.load('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len.npy', allow_pickle=True).item()
    else:
        data, label, category_len = read__data(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
        np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/data', data)
        np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/label', label)
        np.save('D:/Pycharm Projects/CODE/data/tmp/Android/model/category_len', category_len)
    if one_label == True:
        label = label[:,0]
    time0 = time.time()
    accuracy_all_list, max_score = baseline_trainTest(data, label, category_len)  #### 训练KNN、RF等传统模型
    time1 = time.time()
    s1 = 0.1  # train_lstm()  #### 训练LSTM模型
    time2 = time.time()
    accuracy_all = 0.2  # train_fcn()  #### 训练 FCN模型
    time3 = time.time()
    check_model()  #### 输出 dict对应的标签
    return accuracy_all_list, (time1 - time0), s1, (time2 - time1), accuracy_all, (time3 - time2)


def control_button(train_folder):
    '''
        Collect training info and define which folder to train
    '''
    # 变更参数
    # train_keyword, train_folder, test_folder, predict_folder, train_tmp, test_tmp, predict_tmp, \
    # train_tmp_test, model_folder, NB_CLASS = generate_configs(train_folders, train_folder)
    # 存储相关信息
    import os
    train_info_file_ = model_folder + train_info_file
    if os.path.exists(train_info_file_):
        os.remove(train_info_file_)
    fid = open(train_info_file_, 'a')
    fid.write('Index,dataSet,totalRunTime,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES,\
        knn_acc,rf_acc,time_tr,lstm_acc,time_lstm,fcn_acc,time_fcn')
    fid.write('\n')
    fid.close()
    start__time = time.time()
    accuracy_all_list, t1, s1, t2, accuracy_all, t3 = main()
    end__time = time.time()
    run_time = end__time - start__time
    fid = open(train_info_file_, 'a')
    str_ = '%s,%.3f,%s,%d,%.2f,%d,%d,%d,%d,' % (train_folder, run_time, NB_CLASS, sample_rate, \
                                                train_data_rate, window_length, batch_size, units, MAX_NB_VARIABLES)
    fid.write(str_)
    # fid.write('Index,dataSet,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES')
    metrix = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (accuracy_all_list[0], accuracy_all_list[1], t1, \
                                                     np.max(s1), t2, \
                                                     np.max(accuracy_all), t3)
    fid.write(metrix)
    fid.write('\n')
    fid.close()
    return


def predict(one_timeWindow_data):
    '''
        Predict Labels Based on Every Single Window Data In Real Time
    '''
    #### Processing
    cols = predict_window
    tmp_window = one_timeWindow_data.reshape([predict_window * window_length, 1])
    numerical_feature_tmp = gauss_filter(tmp_window, sigma)  # gauss filter
    # fft__window = fft_transform(tmp_window)  # fft process
    fft__window = fft_transform(numerical_feature_tmp)  # fft process

    #### Apply Models
    pca = joblib.load(model_folder + "PCA.m")
    min_max = joblib.load(model_folder + "Min_Max.m")
    # print(fft__window)
    fft__window = pca.transform(fft__window.T)
    # fft__window = fft__window.reshape([len(fft__window), 1])
    # print('Hello There', numerical_feature_tmp.shape, fft__window.shape)

    ''' Changes Here '''
    if use_feature == 'FFT':
        data_window = fft__window.T
    else:
        print(numerical_feature_tmp.shape, fft__window.shape)
        data_window = np.vstack((numerical_feature_tmp, fft__window.T))

    # print(data_window.shape)
    train_max, train_min = float(min_max.data_max_), float(min_max.data_min_)
    test_max, test_min = np.max(data_window), np.min(data_window)
    all_max = train_max if train_max >= test_max else test_max
    all_min = train_min if train_min <= test_min else test_min
    data_window = (data_window - all_min) / (all_max - all_min)

    #### Predict on random forest
    file_write = model_folder + 'newminmax_best_model'
    machine_learning_model = joblib.load(file_write)

    # print(data_window.shape)
    predict_values = machine_learning_model.predict(data_window.T)  # ## If necessary just .T  # now do the final test
    # print(predict_values)
    label_encoder = check_model()  #### 输出 dict对应的标签
    class_list = label_encoder.classes_
    print(class_list)

    # Write for output
    class_list = [u"玩游戏", u"听音乐", u"无操作", u"刷网页", u"看视频", u"写文档"]

    print("用户当前在: \t", class_list[int(predict_values[0])], '\n')
    return


def dl_predict(time_step_data, save_model):
    '''
        Predict Labels Based on Every time step Data In Real Time
    '''
    #### Processing
    tmp_window = time_step_data.T
    numerical_feature_tmp = gauss_filter(tmp_window, sigma)  # gauss filter
    # fft__window = fft_transform(tmp_window)  # fft process

    fft__window = fft_transform(numerical_feature_tmp)  # fft process
    # fft__window = fft__window[1:] #舍弃零频分量
    #### Apply Models
    pca = joblib.load(model_folder + "PCA.m")
    min_max = joblib.load(model_folder + "Min_Max.m")
    # print(fft__window)
    fft__window = pca.transform(fft__window.T)
    # print('Hello There', numerical_feature_tmp.shape, fft__window.shape)

    ''' Changes Here '''
    if use_feature == 'FFT':
        data_window = fft__window.T
    else:
        print(numerical_feature_tmp.shape, fft__window.shape)
        data_window = np.vstack((numerical_feature_tmp, fft__window.T))

    # print(data_window.shape)
    # train_max, train_min = float(min_max.data_max_), float(min_max.data_min_)
    # test_max, test_min = np.max(data_window), np.min(data_window)
    # all_max = train_max if train_max >= test_max else test_max
    # all_min = train_min if train_min <= test_min else test_min
    # data_window = (data_window - all_min) / (all_max - all_min)
    # data_window = data_window.reshape(1, saved_dimension_after_pca)

    data_window = min_max_scaler(data_window)
    total_data = data_window
    total_data = total_data.reshape(1, time_step, saved_dimension_after_pca // time_step)
    #### Deep learning model
    predict_values = save_model.predict(total_data)

    max_label = np.argmax(predict_values)
    label_encoder = check_model()  #### 输出 dict对应的标签
    class_list = label_encoder.classes_
    print(class_list)

    # Write for output
    class_list = [u"玩游戏", u"听音乐", u"无操作", u"刷网页", u"看视频", u"写文档"]
    print("用户当前在: \t", class_list[int(max_label)], '\n')
    return


if __name__ == '__main__':
    control_button('Android')
