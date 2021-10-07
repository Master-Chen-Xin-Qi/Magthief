# coding:utf-8
'''  
    input and preprocess。
    Train 包括完整的train evaluation test, 
    test 指的是完全相同的数据类型代入计算，
    predict指的是没有标签。 
'''

dict_all_parameters = {}
train_info = {}
train_folders = {
            'apps':['offline_video', 'iqiyi', 'word', 'netmusic', 'surfing', 'live'],
            'devices':['hp', 'mac', 'shenzhou', 'windows'],
            'users':['word_1', 'word_2', 'word_3'],
            'History_data':['safari_surfing', 'word_edit', 'safari_youku_video', 'word_scan'],
            '0912':['safari_surfing', 'safari_youku', 'word', 'zuma', 'netmusic'],
            '0913':['netmusic', 'NoLoad', 'safari_surfing', 'tencent_video', 'word_edit', 'Zuma'],
            'Android':['game','music','static','surf','video','work'],
            }

# 采样
sample_rate = 1  # 单位是毫秒 ，>=1
epochs, n_splits = 2 , 10  # 10折交叉验证和epoch数量固定
train_batch_size = 100

########### 处理   #################################################
time_step = 4
M = 400
predict_window = 2  ################ 'Important'
window_length = M//predict_window  # [2, 5, 10, 20, 100]  # 窗口大小
saved_dimension_after_pca, sigma = 150 , 1
use_gauss, use_pca, use_fft = True, True, True  # True
use_time_and_fft = False #False则为频域信号
whether_shuffle_train_and_test = True
use_feature = 'FFT'  # 'Combine, FFT'
########### 处理   #################################################


# 训练
test_ratio, evaluation_ratio = 0.1, 0.1  # 划分训练、测试、验证集
batch_size = 20  # [2, 5, 10]  # 训练 batch大小
units = 1  # [20, 10, 50, 200]  # int(MAX_NB_VARIABLES / 2)
load_already_min_max_data = False #True则载入已经按每行归一化的数据
one_label = False #True只有操作类型标签，False则有操作类型和app两个标签
# 循环遍历
train_data_rate = 0.5  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
i = 0
# 参数
overlap_window = int(0.05 * window_length)  # 窗口和滑动大小
MAX_NB_VARIABLES = window_length * 2
MAX_NB_VARIABLES = (window_length + saved_dimension_after_pca) if use_pca else window_length * 2

### 此处添加文件相关信息 ###
train_info_file = 'train_info_all.txt'
checkpoint_path = "/Users/macintosh/Desktop/CODE/mlp/best_model/best_model.pkl"
# CNN-LSTM参数
APP_CLASS = 12
NB_CLASS = 6
evaluation_ratio = 0.2
train_keyword = ['game', 'music', 'static', 'surf', 'video', 'work']
#train_keyword = ['game', 'music',  'surf']
app_keyword = ['static', 'wechat', 'wps', 'snake', 'paoku', 'net', 'qq', 'hupu', 'weibo', 'quak', 'aiqiyi',  'bilibili']
train_or_test = 'train' #train采取训练数据,test为测试数据
train_folder = 'D:/Pycharm Projects/CODE/data//input//Android'
test_folder = 'D:/Pycharm Projects/CODE/data//input//liangdu//liangdu25'
predict_folder = '../data//input//Android/'
train_tmp = '../data//tmp/Android//tmp/train/'
test_tmp = '../data//tmp/Android//tmp/test/'
predict_tmp = '../data//tmp/Android//tmp/predict/'
train_tmp_test = '../data//tmp/Android//tmp/train/test/'
model_folder = 'D:/Pycharm Projects/CODE/data/tmp/Android/model/'

