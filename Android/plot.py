# coding:utf-8

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rc('font', family='Helvetica')
cmap = plt.cm.jet  # winter

######  文件相关设置 ,更改为自动切换数据集
base_input = '../data/input/'
base_output = 'C:/Users/jhh/Desktop/History_data/explore/'

ML = 10
sigma = 50
M = 4  # 绘制几张子图
linewidth = 5

use_package = '0914'
use_gauss = True
sample_rate = 10

ML = 10
M = 4  # 绘制几张子图
linewidth = 1
NN = 100  # 50ms 绘图取样一次
fontsize = 25
fig_size = (11, 8)

##################################### ##################################
#####################     接收用户参数，决定文件夹等           #########################
##################################### ##################################

######  接收用户参数，决定文件夹等
def interact_With_User():

    import os
    
    key_ = use_package
    
    train_keyword = {
            'apps':['offline_video', 'iqiyi', 'word', 'netmusic', 'surfing', 'live'],
            'devices':['hp', 'mac', 'shenzhou', 'windows'],
            'users':['word_1', 'word_2', 'word_3'],
            'History_data':['safari_surfing.txt', 'word_edit.txt', 'safari_youku_video.txt', 'word_scan.txt'],
            '0912':['safari_surfing_0912.txt', 'safari_youku_0912.txt', 'word_scan_0912.txt', 'zuma_0912.txt', 'netmusic_0912.txt'],
            '0913':['netmusic_0913.txt', 'NoLoad_0913.txt', 'safari_surfing_0913.txt',
                    'tencent_video_0913.txt', 'word_edit_0913.txt', 'Zuma_0913.txt'],
            '0914':['tencent_video.txt']
    }
    
    print('\n--------------------------------------------\n')
    print('keys to use: \n', list(train_keyword.keys()))
    print('\n--------------------------------------------\n')
    # key_ = input('choose your command: use above keys: \n')
    # print('\n--------------------------------------------\n')

    folder = base_input + '/' + str(key_) + '/'
    image_folder = base_output + '/' + str(key_) + '/'
    files_train = train_keyword[key_]
        
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    return folder, image_folder, files_train

folder, image_folder, files_train = interact_With_User()

#######################################################################
######################         读取数据                    ############################
#######################################################################

def read_txt():
        
    from scipy.ndimage import filters
    file_all = []
    
    ######  读单个文件
    def read_(fid):
        read_array = np.loadtxt(fid, skiprows=0) 
        r, c = read_array.shape
        resample_loc = range(int(r / sample_rate))  # Resample
        read_array = read_array[resample_loc, :]  # / 65536
        print('采样前： %d * %d, 采样后： %d * %d' % (r, c, read_array.shape[0], read_array.shape[1]))
        read_array = read_array.reshape([read_array.shape[0] * read_array.shape[1], 1]).T       
        gaussian_X = filters.gaussian_filter1d(read_array, sigma) if use_gauss  else read_array  
        return  gaussian_X

    ######  借用程序，因为不信任reshape的过程，只好list合并
    def vstack_list(tmp):
        if len(tmp) > 1:
            data = np.vstack((tmp[0], tmp[1]))
            for i in range(2, len(tmp)):
                data = np.vstack((data, tmp[i]))
        else:
            data = tmp[0]    
        return data
    
    ######  读所有文件
    for file_ in files_train:
        print('\t正在处理文件：\t %s' % file_)
        fid = open(folder + file_, 'r')
        clean_file = read_(fid)
        file_all.append(clean_file)
        
    ###### 处理成pandas格式
    # file_all = np.array(file_all).reshape([len(file_all[0]), len(file_all)])
    file_all = vstack_list(file_all).T
    file_all = pd.DataFrame(file_all)
    file_all.columns = files_train
    
    return file_all

##################################### ##################################
###########################  进行数据探索   ##################################
##################################### ##################################

def plot(file_, values, i):
    
    indexList = np.array(list(range(int(len(values))))) / 10
    
    ###### 直接绘图
    yDown, yUp = np.min(values), np.max(values)
    
    name = file_.split('.')[0]
    if '-' in name:
        name = name.split('-')[1]

    
    fig = plt.figure(figsize=(100, 20))  # (figsize=(100, 20))    
    # plt.title('Magnetic signals of %s' % name, fontsize=30)  
    ax = fig.add_subplot(111) 
    # plt.xticks([])
    # plt.yticks([])
    
    #     frame = plt.gca()
    #     # y 轴不可见
    #     frame.axes.get_yaxis().set_visible(False)
    #     # x 轴不可见
    #     frame.axes.get_xaxis().set_visible(False)

    # plt.axis('off')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # fig.set_size_inches(24, 14)  # 18.5, 10.5
    plt.axis([0, len(indexList) / 10, yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
    plt.xlabel("Sample Time (second) ", size=24)
    plt.ylabel("Magnetic Signal (uT)", size=24)  
    ax.plot(indexList, values, 'b-', linewidth=linewidth)
    # ax.legend(loc='best')
    plt.savefig(image_folder + '%s%d.png' % (file_, i))
    plt.show()
    print('saved to %s' % (image_folder + file_))

    return

def plot_new(file_, values, i):
    
    indexList = np.array(list(range(int(len(values))))) / 10
    
    ###### 直接绘图
    yDown, yUp = np.min(values), np.max(values)
    
    name = file_.split('.')[0]
    if '-' in name:
        name = name.split('-')[1]

    
    fig = plt.figure()  # (figsize=(100, 20))  # (figsize=(100, 20))    
    # plt.title('Magnetic signals of %s' % name, fontsize=30)  
    ax = fig.add_subplot(111) 
    # plt.xticks([])
    # plt.yticks([])
    
    #     frame = plt.gca()
    #     # y 轴不可见
    #     frame.axes.get_yaxis().set_visible(False)
    #     # x 轴不可见
    #     frame.axes.get_xaxis().set_visible(False)

    # plt.axis('off')

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.set_size_inches(fig_size)  # 18.5, 10.5
    
    #### 认真标定 x 轴数据
    count_line = len(values)
    window_length = int(NN)  # # 相邻数据点的时间间隔
    total_time = int(count_line * NN)  # # 总的时间长度(ms)
    print(count_line, window_length, total_time)
    X = np.ceil(np.array(list(range(0, total_time, window_length))) / 1000)
    print(X)
    
    print(X, values)
    
    maxMI__ = int(np.max(np.max(values)))
    minMI__ = int(np.min(np.min(values)))
    
    min__ = np.min(np.min(values))
    max__ = np.max(np.max(values))
    
    tmp = [float('%.1f' % min__)]
    
    y_tick = list(range(minMI__ + 1, maxMI__, 3))
    tmp.extend(y_tick)
    tmp.append(float('%.1f' % max__))
    
    y_tick = np.array(tmp) 
    m = np.max(y_tick)
    n = np.min(y_tick)
    tt = float('%.1f' % ((m - n) / 5))
    y_tick = np.arange(n, m + 0.5 * tt, tt)        
    print(y_tick)
    
    plt.axis([0, np.ceil(total_time / 1000), yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
    plt.xlabel("Sample Time (second) ", fontsize=fontsize)
    plt.ylabel("Magnetic Signal (mT)", fontsize=fontsize)  
    plt.xticks([50, 100, 150, 200])
    plt.yticks(y_tick)
            
    ax.plot(X, values, 'b-', linewidth=linewidth)
    # ax.legend(loc='best')
    plt.savefig(image_folder + '%s.png' % (file_.split('.')[0]))
    # plt.show()
    print('saved to %s' % (image_folder + file_.split('.')[0]))

    return

def data_explore(file_all):
    
    ######  归一化
    print('\n文件最大值: \n', np.max(file_all), '\n文件最小值: \n', np.min(file_all), '\n')
    print('\n全局最大值: ', np.max(np.max(file_all)), '\n全局最小值: ', np.min(np.min(file_all)))
    
    # file_all = np.max(np.max(file_all)) + np.min(np.min(file_all)) - file_all
    # file_all = (file_all - np.min(np.min(file_all))) / (np.max(np.max(file_all)) - np.min(np.min(file_all)))
    print(file_all.describe())
    
    ######  探索
    print('\n数据探索滤波后结果:\n', file_all.describe())
    
    # file_all.plot()
    # plt.show()

    ######  绘图
    
    for file_ in files_train:
        
        if not use_gauss:
            values = file_all[file_]
            plot(file_, values, 1)
            
            slide = int(len(values) / M)
            print(len(values), slide)
            for i in range(M):
                print(i)
                tmp = values.iloc[i * slide : (i + 1) * slide]
                # plot('filtered', tmp, i)
                plot_new('filtered', tmp, i)

            
        else:
            
            values = file_all[file_]
            plot(file_ + 'filted_all', values, 1)
            
    return

##################################### ##################################
#######################     预处理和准备数据             #############################
##################################### ##################################



if __name__ == '__main__':
    
    file_all = read_txt()
    data_explore(file_all)
