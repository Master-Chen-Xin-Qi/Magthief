# coding:utf-8
'''

'''

import pickle

##### 加载参数，全局变量
with open('D:/Pycharm Projects/CODE/Android/config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    
    dict_all_parameters = pickle.load(f)

    train_tmp = dict_all_parameters['train_tmp']
    test_ratio = dict_all_parameters['test_ratio'] 
    epochs = dict_all_parameters['epochs'] 
    n_splits = dict_all_parameters['n_splits'] 
    model_folder = dict_all_parameters['model_folder'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    NB_CLASS = dict_all_parameters['NB_CLASS']
        
import time
from sklearn import metrics, svm
import joblib
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
import numpy as np

def validatePR(prediction_y_list, actual_y_list):

    ''' 
    
        function : make 3 dictionaries to store and analysis predict result
        
        usage:  prediction_y_list -> label list predicted
                actual_y_list -> label list from original data
                
        return: 
                P ->   precision  , the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
                       true positives and ``fp`` the number of false positives. 
                R ->   recall score,the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
                       true positives and ``fn`` the number of false negative. 
                F1Score -> F-measure means (A + 1) * precision * recall / (A ^ 2 * precision + recall),when A = 1
                       F1Score becomes 2*P*R / P+R
                Micro_average -> average of all ctgry F1Scores
                accuracy_all -> accuracy means 'TP+TN / TP+TN+FP+FN'
                
    '''
    right_num_dict = {}
    prediction_num_dict = {}
    actual_num_dict = {}

    Precise = {}
    Recall = {}
    F1Score = {}
    
    if len(prediction_y_list) != len(actual_y_list):
        raise(ValueError)    
    
    for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
        
        if p_y not in prediction_num_dict:
            prediction_num_dict[p_y] = 0
        prediction_num_dict[p_y] += 1

        if a_y not in actual_num_dict:  # here mainly for plot 
            actual_num_dict[a_y] = 0
        actual_num_dict[a_y] += 1

        if p_y == a_y:  # basis operation,to calculate P,R,F1
            if p_y not in right_num_dict:
                right_num_dict[p_y] = 0
            right_num_dict[p_y] += 1
    
    for i in  np.sort(list(actual_num_dict.keys()))  : 
                
        count_Pi = 0  # range from a to b,not 'set(list)',because we hope i is sorted 
        count_Py = 0
        count_Ri = 0
        count_Ry = 0

        for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
            
            
            if p_y == i:
                count_Pi += 1
                
                if p_y == a_y:                              
                    count_Py += 1
                    
            if a_y == i :
                count_Ri += 1
                
                if a_y == p_y:
                    count_Ry += 1    
        
        Precise[i] = count_Py / count_Pi if count_Pi else 0               
        Recall[i] = count_Ry / count_Ri if count_Ri else 0
        F1Score[i] = 2 * Precise[i] * Recall[i] / (Precise[i] + Recall[i]) if Precise[i] + Recall[i] else 0
    
    Micro_average = np.mean(list(F1Score.values()))
    
    lenL = len(prediction_y_list)
    sumL = np.sum(list(right_num_dict.values()))
    accuracy_all = sumL / lenL
        
    return Precise, Recall, F1Score, Micro_average, accuracy_all

def naive_bayes_classifier(trainX, trainY, train_wt):  # Multi nomial Naive Bayes Classifier
    
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(trainX, trainY)
    
    return model

def knn_classifier(trainX, trainY, train_wt):  # KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(trainX, trainY)
    return model

def logistic_regression_classifier(trainX, trainY, train_wt):  # Logistic Regression Classifier
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(trainX, trainY, train_wt)
    
    return model

def random_forest_classifier(trainX, trainY, train_wt):  # Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=30)
    model.fit(trainX, trainY, train_wt)
    
    return model

def decision_tree_classifier(trainX, trainY, train_wt):  # Decision Tree Classifier
    
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(trainX, trainY, train_wt)
    
    return model
 
def gradient_boosting_classifier(trainX, trainY, train_wt):  # GBDT(Gradient Boosting Decision Tree) Classifier
    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(trainX, trainY, train_wt)
    
    return model

def ada_boosting_classifier(trainX, trainY, train_wt):  # AdaBoost Ensemble Classifier
    
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(trainX, trainY, train_wt)
    
    return model

def svm_classifier(trainX, trainY, train_wt):  # SVM Classifier

    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(trainX, trainY, train_wt)
    
    return model

def svm_cross_validation(trainX, trainY):  # SVM Classifier using cross validation

    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(trainX, trainY)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(trainX, trainY)
    
    return model

def split_train_Validate_test(data, vali_prcnt, test_prcnt):
    
    ''' 
        function: cut data
                  Cut data into train,predict,feather,label ,twice
        usage:  
                data --> array 
                target --> array label
                random_state --> will random shuffle
                can use ,stratify = ('stratify') here
    ''' 
    rs = 10
    target = data[:, -1]
    value = data[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(value, target, test_size=vali_prcnt, random_state=rs, shuffle=whether_shuffle_train_and_test)  # default 0.2
    X_valdt, X_test, y_valdt, y_test = train_test_split(X_test, y_test, test_size=test_prcnt, random_state=rs, shuffle=whether_shuffle_train_and_test)
    
    return X_train, X_test, y_train, y_test, X_valdt, y_valdt

def train_test_evalation_split(data, label):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0)
    return X_train, X_test, y_train, y_test 

def baseline_trainTest():  


    """   Load the data   """
    
    data = np.loadtxt(train_tmp + 'After_pca_data.txt')
    label = np.loadtxt(train_tmp + 'After_pca_label.txt')

    ######  此处将对train 
    ######  10折交叉验证改为
    ######  对所有数据交叉验证，然后取均值作为模型评分
    
    X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label)
    X = X_train
    y = y_train
    # X = data
    # y = label

    print('data shape: \n', data.shape)

    file_write = model_folder + 'best_model'
    model_save_file = file_write
    model_save = {}
    train_wt = None
    
    num_train = X.shape[0]
    is_binary_class = (len(np.unique(y)) == 2)
    
    # test_classifiers = ['NB','KNN', 'LR', 'RF', 'DT','SVM','GBDT','AdaBoost']
    test_classifiers = ['KNN', 'RF']  # , 'DT','LR']  # , 'GBDT', 'AdaBoost']

    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier,
                 'AdaBoost':ada_boosting_classifier
    }
        
    print ('******************** Data Info *********************')
    
    scores_Save = []
    model_dict = {}
    accuracy_all_list = []
    
    for classifier in test_classifiers:
        
        print ('******************* %s ********************' % classifier)
        
        scores = []
        
        skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        # print(skf_cv) 
        i = 0
        for train_index, test_index in skf_cv.split(X, y):
        
            i += 1                                               
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = classifiers[classifier](X_train, y_train, train_wt)
            predict_y = model.predict(X_test) 
            Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y, y_test) 
            print(accuracy_all, F1Score)
            # print (' \n Precise: %f \n' % Precise, 'Recall: %f \n' % Recall, 'F1Score: %f \n' % F1Score)  # judge model,get score
        
            scores.append({'cnt:':i, 'mean-F1-Score':Micro_average, 'accuracy-all':accuracy_all})
            accuracy_all_list.append(accuracy_all)

        Micro_average, accuracyScore = [], []
        
        for item in scores:
            Micro_average.append(item['mean-F1-Score'])
            accuracyScore.append(item['accuracy-all'])
            
        Micro_average = np.mean(Micro_average)
        accuracyScore = np.mean(accuracyScore)
                
        scoresTmp = [accuracy_all, Micro_average]
        print (' \n accuracy_all: \n', accuracy_all, '\nMicro_average:  \n', Micro_average)  # judge model,get score
        
        scores_Save.append(scoresTmp)
        model_dict[classifier] = model 

    print ('******************* End ********************')
    
    scores_Save = np.array(scores_Save)
    
    max_score = np.max(scores_Save[:, 1])
    index = np.where(scores_Save == np.max(scores_Save[:, 1]))
    index_model = index[0][0]
    model_name = test_classifiers[index_model]
    
    print (' \n Best model: %s \n' % model_name, '\n Best model score: \n', max_score)
    
    joblib.dump(model_dict[model_name], file_write)

    ######## 重新调整，打印混淆矩阵
    model_sort = []
    scores_Save1 = scores_Save * (-1)
    sort_Score1 = np.sort(scores_Save1[:, 1])  # inverse order
    for item  in sort_Score1:
        index = np.where(scores_Save1 == item)
        index = index[0][0] 
        model_sort.append(test_classifiers[index])
         
    #### 使用全部数据，使用保存的，模型进行实验
    model = model_dict[model_name]       
    predict_y_left = model.predict(X_test_left)  # now do the final test
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y_left, y_test_left) 
    print ('\n final test: model: %s, F1-mean: %f,accuracy: %f' % (model_sort[0], Micro_average, accuracy_all))
 
    s1 = metrics.accuracy_score(y_test_left, predict_y_left)
    f2 = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'traditional_train_test_confusion_matrix.csv', f2.astype(int), delimiter=',', fmt='%d')
    # f1 = metrics.fbeta_score(y_test_left, predict_y_left,beta= 0.5)
    print ('Not mine: final test: model: %s,\n accuracy: %f' % (model_sort[0], s1), 'matrix:\n', f2.astype(int))

    return  accuracy_all_list, max_score

if __name__ == '__main__':

    baseline_trainTest()
