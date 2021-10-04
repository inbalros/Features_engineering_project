import numpy as np
import pandas as pd
#import auto_by_criteria as ac
#from automatic_FE.importances_features import *
import matplotlib.pyplot as plt
import sys
import os
print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

def encode_categorical(data,var_mode):
    le = LabelEncoder()
    for i in var_mode:
        data[i] = le.fit_transform(data[i].astype(str))


def find_X_from_RF(train,test,x_names,y_names):
    un_train, _ = np.unique(pd.Series.tolist(train[y_names]), return_counts=True)
    print("train un: " + str(un_train) )

    central_clf = RandomForestClassifier(n_estimators=100, random_state=None) \
                .fit(train[x_names], np.array(train[y_names]))

    criterion_trees=0
    n_leaves_all =0
    max_depth_all=0
    node_count_all=0
    result_quality_cur = 0

    #if index_trees != number_of_trees:          ####check this####
    #    global criterion_base
    #    pred_acc =0
    #    criterion_trees = criterion_base
    #else:
    prediction = central_clf.predict(test[x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    pred_acc = acu_test

    #confusion_all += confusion_matrix(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)

    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)

    print("true un: "+str(un_true)+" pred un: "+str(un_pred))
    print("acc: " + str(pred_acc))

    only_one = 0
    #y_true.append(0)
    #y_true.append(1)
    #y_pred.append(0)
    #y_pred.append(1)
    #y_true.append(0)
    #y_true.append(1)
    #y_pred.append(1)
    #y_pred.append(0)
        #print("zero or ones")
    precision_all = metrics.precision_score(y_true, y_pred,average='weighted',zero_division=0)
    recall_all = metrics.recall_score(y_true, y_pred,average='weighted',zero_division=0)
    f_measure_all = metrics.f1_score(y_true, y_pred,average='weighted')

    print("f: "+str(f_measure_all))

    #try:
    if len(un_true) != 1 and len(un_pred) != 1:
        #roc_all = metrics.roc_auc_score(y_true, y_pred,multi_class="ovr")
        #print("roc: "+str(roc_all))

    #except:
    #    print("exception_roc")
    #    roc_all =  0
    #try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred,pos_label=un_true[0])
        prc_all = metrics.auc(recall_prc, precision_prc)

        print("prc: "+str(prc_all))
    #except:
    #    print("exception_prc - N")
    #    prc_all = 1

    ############## All the criterion options on cur_tree:


    for cur_tree in central_clf.estimators_:
        n_leaves_all += cur_tree.tree_.n_leaves
        max_depth_all += cur_tree.tree_.max_depth
        node_count_all += cur_tree.tree_.node_count
    n_leaves_all /= 100
    max_depth_all /= 100
    node_count_all /= 100
    criterion_trees /= 100
    print("n_leaves_all: "+str(n_leaves_all))
    print("max_depth_all: "+str(max_depth_all))
    print("node_count_all: "+str(node_count_all))



    #return (np.array(list((pred_acc,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))), central_clf,only_one)


def try_prediction(data,X_names,y_names):
    kfold = KFold(3)
    # kfold = KFold(3)
    for train, test in kfold.split(data):
        data_train = (data.iloc[train]).copy().reset_index(drop=True)
        data_test = (data.iloc[test]).copy().reset_index(drop=True)
        find_X_from_RF(data_train, data_test, X_names, y_names)

    #return arr
    # dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred


def categ(data):
    print("DB size: "+ str(data.shape))
    print(data.dtypes)
    try_data = data.select_dtypes(include='object')
    category_columns = try_data.columns
    print(category_columns.shape)
    print(data.head())
    print(data.describe().T)
    print(try_data.isnull().sum())
    print('the value counts of the target are:')
    print(data.iloc[:, -1].value_counts())

    encode_categorical(data,category_columns)
    print(data.head())
    print(data.describe().T)
    print(try_data.isnull().sum())

    null_data = try_data[try_data.isnull().any(axis=1)]
    print(null_data.shape)
    print('the value counts of the target are:')
    print(data.iloc[:, -1].value_counts())
    data.iloc[:, -1].value_counts().plot(kind='bar')
    plt.show()
    try_prediction(data, list(data.iloc[:, :-1].columns), str(list(data.iloc[:,-1 :].columns)[0]))


#    data_path = os.path.join(one_back,r'Data\magic04\magic.csv')


















exit(0)

print("start - magic04")
#2) data set = magic04 https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
data_path = os.path.join(one_back,r'Data\magic04\magic.csv')
magic = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,12)])
X_names_ds = ["att"+str(i) for i in range(1,11)]
y_names = "att11"
categ(magic)
un_class = len(magic[y_names].unique())
print(un_class)


print("start - skin_nonSkin")
# 3) data set = skin_NonSkim https://archive.ics.uci.edu/ml/datasets/skin+segmentation#
data_path = os.path.join(one_back,r'Data\skin_NonSkin\skin_NoSkin.csv')
skin = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,5)])
X_names_ds = ["att"+str(i) for i in range(1,4)]
y_names = "att4"
categ(skin)
un_class = len(skin[y_names].unique())
print(un_class)

'''
print("start - cars")
# 4) data set = car https://archive.ics.uci.edu/ml/machine-learning-databases/car/
data_path = os.path.join(one_back,r'Data\car\car.csv')
cars = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
X_names_ds = ["att"+str(i) for i in range(1,7)]
y_names = "att7"
categ(cars)
un_class = len(cars[y_names].unique())
print(un_class)
'''

print("start - abalone")
#5) data set = abalone : https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names
data_path = os.path.join(one_back,r'Data\abalone\abalone.csv')
abalone = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,10)])
X_names_ds_all = ["att"+str(i) for i in range(2,10)]
X_names_ds = ["att"+str(i) for i in range(2,9)]
abalone = abalone[X_names_ds_all]
y_names = "att9"
categ(abalone)
un_class = len(abalone[y_names].unique())
print(un_class)



print("start - bank")
# 6) data set = bank  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
data_path = os.path.join(one_back,r'Data\bank\bank.csv')
bank = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names_ds = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
categ(bank)
un_class = len(bank[y_names].unique())
print(un_class)


print("start - wine")
# 7) data set = wine  https://archive.ics.uci.edu/ml/datasets/Wine+Quality
data_path = os.path.join(one_back,r'Data\wine\winequality-white.csv')
wine = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,13)])
X_names_ds = ["att"+str(i) for i in range(1,12)]
y_names = "att12"
categ(wine)
un_class = len(wine[y_names].unique())
print(un_class)



print("start - blood")
# 8) data set = blood  https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
data_path = os.path.join(one_back,r'Data\blood_transfusion\blood.csv')
blood = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,6)])
X_names_ds = ["att"+str(i) for i in range(1,5)]
y_names = "att5"
categ(blood)
un_class = len(blood[y_names].unique())
print(un_class)

print("start - wifi")
# 9) data set = wifi  https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
data_path = os.path.join(one_back,r'Data\wifi\wifi.csv')
wifi = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,9)])
X_names_ds = ["att"+str(i) for i in range(1,8)]
y_names = "att8"
categ(wifi)
un_class = len(wifi[y_names].unique())
print(un_class)


print("start - chess")
# 10) data set = chess https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29
data_path = os.path.join(one_back,r'Data\chess\chess.csv')
chess = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
X_names_ds = ["att"+str(i) for i in range(1,7)]
y_names = "att7"
categ(chess)
un_class = len(chess[y_names].unique())
print(un_class)



print("start - letters")
#1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
data_path = os.path.join(one_back,r'Data\letter_recognition\letter_recognition.csv')
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names_ds = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
categ(letters)
un_class = len(letters[y_names].unique())
print(un_class)


exit(0)

