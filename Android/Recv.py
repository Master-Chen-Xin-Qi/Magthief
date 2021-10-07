# coding:utf-8
'''
代码功能：接收移动设备的实时磁信号
'''
import socket, time, os
import pandas as pd, numpy as np

from CODE.Android.config import predict_window, train_folder, window_length, use_feature
from CODE.Android.Model import plot
from CODE.Android.main import predict
from mlp.mlp_solver import Solver
import math
saveTime = 3  # Minutes

#
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
                a = Solver().predict(acc_data, "../mlp/best_model/best_model.pkl")
                one_window_data = one_window_data-a
                ######################
                predict(one_window_data)
                sock.close()
                return one_window_data
                break

def real_time():
    t = 0
    while True:
        import time
        t += 5
        # time.sleep(2)
        one_window_data = real_time_processors()
        plot('%d' % t, one_window_data)
    return


if __name__ == '__main__':

    # Collect
    data_save = '_zhijia.txt'
    collect_mag(data_save)

    # Predict 
    #real_time()
