import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  KFold
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
import itertools
from sklearn.metrics import pairwise_distances_argmin_min
import math
import time
from sklearn import metrics

import featuretools as ft


#############################
#                           #
#  data sets - to evaluate  #
#                           #
#############################

db_name = r'D:\phd\DB\Diabetes.csv'
dfAllPred= pd.DataFrame(
        columns=['dataset_name','method','number_of_classes','clusterNumber','dataset_size','using_model', 'fit_time', 'pred_time','train_size_per_fold','test_size_per_fold', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])

writerResults = pd.ExcelWriter('results.xlsx')

def write_to_excel():
    dfAllPred.to_excel(writerResults,'Cluster_Decomposition')
    writerResults.save()


'''
dataset = pd.read_csv(db_name).drop('class_word',1)
label_bugs = ''
print(dataset)

cur_model = RandomForestClassifier(n_estimators=1000, max_depth=5)
one_ba_model = cur_model.fit(dataset.drop(label_bugs, axis=1), dataset[label_bugs])
pred = one_ba_model.predict(x_test)
'''
def encode_categorial(data):
    """
    change the category data to numbers represent the value
    :param data:
    :return:
    """
    le = LabelEncoder()
    for col in data:
        # categorial
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
def multi_class_DT_pred(train,test,data, x_names,y_names):
    start_time = time.time()
    central_clf = DecisionTreeClassifier(max_depth=3).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    prediction = central_clf.predict(data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    predictionOnTrain = central_clf.predict(data.iloc[train][x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
    train_size = len(train)
    test_size = len(test)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    return np.array(list((fit_time, pred_time,train_size,test_size, acu_test, acu_train, precision, recall, fscore)))

def normalize_results(pred,index,dataset_name,method,number_of_classes,cluster_number,dataset_size,using_model):
    pred /= index
    pred_list = pred.tolist()
    pred_list = [dataset_name, method, number_of_classes, cluster_number, dataset_size, using_model] + pred_list
    return pred_list

def predict_kfold(name,data, x_names,y_names,number_of_classes,paramK=5):
    kfold = KFold(paramK, True)
    index = 0
    size = data.shape[0]
    all_OVO_pred = 0
    all_multi_class_DT_pred =0
    for train, test in kfold.split(data):
        index += 1
        all_multi_class_DT_pred += multi_class_DT_pred(train,test,data, x_names,y_names)

    results_all_multi_class_DT_pred = normalize_results(all_multi_class_DT_pred,index,name,"decision_tree",number_of_classes,"-",size,"decision_tree")

    dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred


print("start - letters")
#1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition.csv'
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
letters['un'] = range(1, len(letters) + 1)
encode_categorial(letters)
un_class = len(letters[y_names].unique())
print(letters)


#############################

let_all = letters.drop(y_names,1)
let_pred = letters[[y_names,'un']]

entities = { "letters_all" : (let_all,'un'), "letters_pred" : (let_pred,'un') }
relationships =[('letters_pred','un','letters_all','un')]

feature_matrix_customers, features_defs = ft.dfs(entities=entities,relationships=relationships,target_entity="letters_pred")
print(feature_matrix_customers)
#############################






predict_kfold("letters", letters, X_names, y_names, un_class, 10)

print("start - zoo")
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\zoo\zoo_R.dat'
zoo = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)],sep="\t")
X_names = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
encode_categorial(zoo)
un_class = len(zoo[y_names].unique())
predict_kfold("zoo", zoo, X_names, y_names, un_class, 10)

write_to_excel()

print("done")


