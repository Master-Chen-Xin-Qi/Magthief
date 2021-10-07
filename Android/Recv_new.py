# coding:utf-8
'''
代码功能：接收移动设备的实时磁信号
'''
import socket, time, os
import pandas as pd, numpy as np

from CODE.Android.config import predict_window, train_folder, window_length, use_feature, time_step, saved_dimension_after_pca
from CODE.Android.Model import plot, dl_plot
from CODE.Android.main import predict, dl_predict
from CODE.Android.main import generate_my_model
import matplotlib.pyplot as plt
from mlp.mlp_solver import Solver
import math
saveTime = 6  # Minutes


def collect_mag(data_save):
    folder_save = train_folder
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    if os.path.exists(folder_save + data_save):
        os.remove(folder_save + data_save)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8090))  # 10.162.117.28
    packages_data = []
    save_packages = 500
    st = time.time()
    count_packages = 0
    while True:
        # 读数
        data, _ = sock.recvfrom(150)
        #print("111")
        one_package = data.decode('ascii')
        one_package = one_package.split(',')
        if len(one_package) >= 9:  # WARNING HERE: Different phones are different
            #print(one_package)
            #one_package = float(one_package[-3])
            one_package = [float(one_package[-3]), float(one_package[-2]),float(one_package[3]),float(one_package[4])]
            packages_data.append(one_package)  # Four info: time, x-mag,y-mag,z-mag
            count_packages += 1
        # 存盘
        if len(packages_data) > save_packages:
            fid = open(folder_save + data_save, 'a')
            packages_data = pd.DataFrame(packages_data)
            packages_data.to_csv(fid, header=False, index=False)
            fid.close()
            packages_data = []
            print('Saving mag data\t', count_packages)
        et = time.time()  
        if et - st > saveTime * 60:
            print('Mag Sensor Used time: %d seconds, collected %d samples' % (et - st, count_packages))
            break
        
        
def real_time_processors():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8090))  # 10.162.2.79
    packages_data = []
    acc_data = []
    st = time.time()
    count_packages = 0
    while True:
        # 读数
        data, _ = sock.recvfrom(150)
        one_package = data.decode('ascii')
        one_package = one_package.split(',')
        #print(one_package)
        if len(one_package) >= 9:  # WARNING HERE: Different phones are different
            acc_package = [float(one_package[2]),float(one_package[3]),float(one_package[4])]
            one_package = float(one_package[-3])
            #one_package = math.sqrt(float(one_package[-3])*float(one_package[-3])+float(one_package[-2])*float(one_package[-2])+float(one_package[-1])*float(one_package[-1]))
            #print(one_package)
            count_packages += 1
            
            if count_packages <= predict_window * window_length:
                packages_data.append(one_package)  # Four info: time, x-mag,y-mag,z-mag
                acc_data.append(acc_package)
                # print(acc_data)
                # print(packages_data)
            else:
                one_window_data = np.array(packages_data).reshape([predict_window * window_length, 1])
                acc_data = np.array(acc_data)
                ######################
                #消除地磁
                #a = Solver().predict(acc_data, "../mlp/best_model/best_model.pkl")
                #one_window_data = one_window_data-a
                ######################
                ###随机森林预测
                predict(one_window_data)
                sock.close()
                return one_window_data
                break


def dl_real_time_processors(model):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8090))  # 10.162.2.79
    packages_data = []
    acc_data = []
    st = time.time()
    count_packages = 0
    while True:
        # 读数
        data, _ = sock.recvfrom(150)
        one_package = data.decode('ascii')
        one_package = one_package.split(',')
        if len(one_package) >= 9:  # WARNING HERE: Different phones are different
            acc_package = [float(one_package[2]), float(one_package[3]), float(one_package[4])]
            one_package = float(one_package[-3])
            count_packages += 1

            if count_packages <= predict_window * window_length:
                packages_data.append(one_package)  # Four info: time, x-mag,y-mag,z-mag
                acc_data.append(acc_package)
                # print(acc_data)
                # print(packages_data)
            else:
                one_window_data = np.array(packages_data).reshape([predict_window * window_length,1])
                acc_data = np.array(acc_data)
                time_step_data = one_window_data

                ######################
                # 消除地磁
                # a = Solver().predict(acc_data, "../mlp/best_model/best_model.pkl")
                # one_window_data = one_window_data-a
                ######################
                ###深度学习预测
                dl_predict(time_step_data, model)
                sock.close()
                count_packages = 0
                return time_step_data
                break
            

def real_time(control, model):
    t = 0
    while True:
        import time
        t += 5
        # time.sleep(2)
        if(control == 'rf'):
            one_window_data = real_time_processors()
            plot('%d' % t, one_window_data)
        if(control == 'dl'):
            time_step_data = dl_real_time_processors(model)
            #x = np.arange(0,window_length*predict_window*time_step/100,0.01)
            dl_plot('%d' % t,time_step_data)

    return


if __name__ == '__main__':
    save_model = None
    # Collect
    data_save = 'demo_surf2_zhijia.txt'
    #collect_mag(data_save)
    
    control = 'dl'#参数为：'rf' 'dl'

    #Predict
    if control == 'dl':
        save_model = generate_my_model()
        save_model.load_weights('/Users/macintosh/Desktop/CODE/data/tmp/Android/model/train_fold6_newdemo_5_30_64lstm_acc_weights.h5')
    real_time(control,save_model)

