import time
from stree import Stree
import numpy as np
import pandas as pd
from sklearn import metrics
#from automatic_FE.job_wrapper import *
import os
import sys
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import KFold

dfbaselinePred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
         'number_of_folds','depth_max','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features','all_features',
         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after','method'])


print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)

dataset_name= "wifi"

#result_path=os.path.join(one_back,  r"results\results_"+dataset_name+".txt")

index3 = 1

number_of_features_cur = 0
number_of_classes_cur = 0
using_criterion_cur=0
number_of_folds_cur=0
number_of_trees_per_fold_cur=0
delete_used_f_cur=0

def set_params(number_of_features,number_of_classes,using_criterion,number_of_folds,number_of_trees_per_fold,delete_used_f):
    global number_of_features_cur
    number_of_features_cur = number_of_features
    global number_of_classes_cur
    number_of_classes_cur = number_of_classes
    global using_criterion_cur
    using_criterion_cur = using_criterion
    global number_of_folds_cur
    number_of_folds_cur=number_of_folds
    global number_of_trees_per_fold_cur
    number_of_trees_per_fold_cur = number_of_trees_per_fold
    global delete_used_f_cur
    delete_used_f_cur =delete_used_f

def addToEXCEL_baseline(acu_test,precision,recall,f_measure,roc,prc,n_leaves,max_depth,node_count,baseline,data,depth,x_names):
    dfbaselinePred.loc[len(dfbaselinePred)] = np.array(
        [dataset_name, number_of_features_cur, number_of_classes_cur, str(data.shape[0]), using_criterion_cur, str(acu_test),
         "before_criterion",
         str(precision), str(recall), str(f_measure), str(roc), str(prc),
         str(n_leaves), str(max_depth), str(node_count),
         number_of_folds_cur, str(depth), number_of_trees_per_fold_cur, "", delete_used_f_cur,
         "",
         str(x_names), "", "",
         "", "", "",
         "", "", "",
         "", "",baseline])
    write_to_excel_df_baseline()


def write_to_excel_df_baseline():
    global index3
    writerResults = pd.ExcelWriter(os.path.join(one_back,  r"results_imp\results_baseline_"+dataset_name+"_"+str(index3)+".xlsx"))
    index3+=1
    dfbaselinePred.to_excel(writerResults,'results')
    writerResults.save()

def _distances(self, node, data) -> np.array:
    """Compute distances of the samples to the hyperplane of the node

        Parameters
        ----------
        node : Snode
            node containing the svm classifier
        data : np.ndarray
            samples to compute distance to hyperplane

        Returns
        -------
        np.array
            array of shape (m, nc) with the distances of every sample to
            the hyperplane of every class. nc = # of classes
    """
    X_transformed = data[:, node._features]
    if self._normalize:
        X_transformed = node._scaler.transform(X_transformed)
    return node._clf.decision_function(X_transformed)


def predict_class(xp, node ,Stree):
    if xp is None:
        return [], []
    #if node.is_leaf():
    # set a class for every sample in dataset
    #    prediction = np.full((xp.shape[0], 1), node._class)
    #    return prediction, indices
    Stree.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = Stree.splitter_.part(indices)
    xp['new'] = 1
    xp.loc[i_u,'new'] = 0
    #xp.iloc[i_u]['new'] = 0

    #df = pd.DataFrame()

    #prx_u, prin_u = predict_class(x_u, i_u, node.get_up())
    #prx_d, prin_d = predict_class(x_d, i_d, node.get_down())
    #return np.append(prx_u, prx_d), np.append(prin_u, prin_d)
    return xp




def number_of_leafs_STree(clf):
    a = str(clf)
    num_of_leafs = (a.count("Leaf"))
    return num_of_leafs

def number_of_nodes_STree(clf):
    a = str(clf)
    num_of_nodes = (a.count("\n"))
    return num_of_nodes

def max_depth_STree(clf):
    a = str(clf)
    max_depth = (max(list((w.count("-") for w in a.split("\n")))))
    return max_depth

from sklearn.svm import SVC, LinearSVC

def baseline_STree_classifier_competition(train,test,data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):
    cur_tree_linear = Stree(max_depth=1,kernel='linear').fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    cur_tree_linear2 = Stree(max_depth=1,kernel='linear').fit(data[["att1","att2"]], np.array(data.iloc[train][y_names]))
    predict_class((data.iloc[train][x_names]).copy().reset_index(drop=True), cur_tree_linear.tree_, cur_tree_linear)

    cur_tree_poly = Stree(max_depth=3,kernel='poly').fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    cur_tree_rbf = Stree(max_depth=3,kernel='rbf').fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    cur_tree_sigmoid = Stree(max_depth=3,kernel='sigmoid').fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))

    cur_tree_linear.tree_._clf.predict(data.iloc[test][x_names])

    start_time = time.time()
    #prediction = cur_tree.predict(data.iloc[test][x_names])  # the predictions labels
    C= 1.0
    kernel= "linear"
    max_iter = 1e5
    random_state = None
    tol= 1e-4
    degree= 3
    gamma = "scale"
    split_criteria = "impurity"
    criterion = "entropy"
    min_samples_split = 0
    max_features = None
    splitter= "random"
    multiclass_strategy = "ovo"
    normalize = False
    new_svc= SVC(
        kernel=kernel,
        max_iter=max_iter,
        tol=tol,
        C=C,
        gamma=gamma,
        degree=degree,
        random_state=random_state,
        probability=True,
        decision_function_shape=multiclass_strategy
    ).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))

    new_svc_pred = new_svc.predict(data.iloc[test][x_names])
    prediction_linear = cur_tree_linear.tree_._clf.predict(data.iloc[test][x_names])
    prediction_poly = cur_tree_poly.tree_._clf.predict(data.iloc[test][x_names])
    prediction_rbf = cur_tree_rbf.tree_._clf.predict(data.iloc[test][x_names])
    prediction_sigmoid = cur_tree_sigmoid.tree_._clf.predict(data.iloc[test][x_names])
    df = pd.DataFrame()
    df['prediction_linear']= prediction_linear
    df['new_svc_pred']=new_svc_pred
    df['prediction_linear_f']= cur_tree_linear.predict(data.iloc[test][x_names])
    df['prediction_poly']= prediction_poly
#    df['prediction_poly_f']= cur_tree_poly.predict(data.iloc[test][x_names])
    df['prediction_rbf']= prediction_rbf
#    df['prediction_rbf_f']= cur_tree_rbf.predict(data.iloc[test][x_names])
    df['prediction_sigmoid']= prediction_sigmoid
#    df['prediction_sigmoid_f']= cur_tree_sigmoid.predict(data.iloc[test][x_names])
    df['REAL'] = pd.Series.tolist(data.iloc[test][y_names])
    end_time = time.time()
    #cur_tree_linear.tree_._clf.predict_proba(data.iloc[test][x_names])
def find_X_from_baseline_STree_train_test(train,test,x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):

    if depth == None:
        cur_tree = Stree(max_depth=100000).fit(train[x_names], train[y_names])
    else:
        cur_tree = Stree(max_depth=depth).fit(train[x_names], train[y_names])
    start_time = time.time()
    prediction = cur_tree.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    pred_acc = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)
    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    if len(un_true) == 1 or len(un_pred) == 1:
        y_true.append(0)
        y_true.append(1)
        y_pred.append(0)
        y_pred.append(1)
        y_true.append(0)
        y_true.append(1)
        y_pred.append(1)
        y_pred.append(0)
        # print("zero or ones")
    criterion_trees = 0
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred, average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        # print("exception_roc")
        roc_all = 0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1

    ############## All the criterion options on cur_tree:
    n_leaves_all = number_of_leafs_STree(cur_tree)
    max_depth_all = max_depth_STree(cur_tree)
    node_count_all = number_of_nodes_STree(cur_tree)

    return np.array(list((pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))).tolist()

from sklearn.preprocessing import LabelEncoder

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
            #print("this is one"+ col)
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
        #print(col)
        #print(data[col].dtype)

'''
print(dataset_name)
if(dataset_name=='magic'):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\magic04\magic.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 12)])
    X_names_ds = ["att" + str(i) for i in range(1, 11)]
    y_names = "att11"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'magic'
    f_number = 10
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 11)]
    label = "att11"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="skin_nonSkin"):
    print(dataset_name)
    data_path =os.path.join(one_back, r'Data\skin_NonSkin\skin_NoSkin.csv')
    choosen_data = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,5)])
    #choosen_data = choosen_data[:100]
    X_names_ds = ["att" + str(i) for i in range(1, 4)]
    y_names = "att4"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'skin_nonSkin'
    f_number = 3
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 4)]
    label = "att4"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="cars"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\car\car.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 8)])
    X_names_ds = ["att" + str(i) for i in range(1, 7)]
    y_names = "att7"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'cars'
    f_number = 6
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 7)]
    label = "att7"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="abalone"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\abalone\abalone.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 10)])
    X_names_ds = ["att" + str(i) for i in range(1, 9)]
    y_names = "att9"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'abalone'
    f_number = 8
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 9)]
    label = "att9"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="bank"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\bank\bank.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 18)])
    X_names_ds = ["att" + str(i) for i in range(1, 17)]
    y_names = "att17"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'bank'
    f_number = 16
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 17)]
    label = "att17"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="wine"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\wine\winequality-white.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 13)])
    X_names_ds = ["att" + str(i) for i in range(1, 12)]
    y_names = "att12"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'wine'
    f_number = 11
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 12)]
    label = "att12"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="blood"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\blood_transfusion\blood.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 6)])
    X_names_ds = ["att" + str(i) for i in range(1, 5)]
    y_names = "att5"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'blood'
    f_number = 4
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 5)]
    label = "att5"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="wifi"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\wifi\wifi.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 9)])
    X_names_ds = ["att" + str(i) for i in range(1, 8)]
    y_names = "att8"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'wifi'
    f_number = 7
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 8)]
    label = "att8"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="chess"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\chess\chess.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 8)])
    X_names_ds = ["att" + str(i) for i in range(1, 7)]
    y_names = "att7"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'chess'
    f_number = 6
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 7)]
    label = "att7"
    data_chosen[label] = choosen_data[label]

elif(dataset_name=="letters"):
    print(dataset_name)
    data_path = os.path.join(one_back,r'Data\letter_recognition\letter_recognition.csv')
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 18)])
    X_names_ds = ["att" + str(i) for i in range(1, 17)]
    y_names = "att17"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'letters'
    f_number = 16
    data_chosen = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 17)]
    label = "att17"
    data_chosen[label] = choosen_data[label]

else:
    exit(0)


index=0
kfold = KFold(5)


for train, test in kfold.split(data_chosen):
    index += 1
    arr_baseline1, clf = baseline_STree_classifier_competition(train, test, data_chosen, X_names_ds, y_names,None, None, 100, 1)
'''







#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');
#ax = plt.gca()
#xlim = ax.get_xlim()
#w = svc.coef_[0]
#a = -w[0] / w[1]
#xx = np.linspace(xlim[0], xlim[1])
#yy = a * xx - svc.intercept_[0] / w[1]
#plt.plot(xx, yy)
#yy = a * xx - (svc.intercept_[0] - 1) / w[1]
#plt.plot(xx, yy, 'k--')
#yy = a * xx - (svc.intercept_[0] + 1) / w[1]
#plt.plot(xx, yy, 'k--')

 #a = str(clf)
 #   num_of_leafs = (a.count("Leaf"))
 #   num_of_nodes = (a.count("\n"))
    # arr = a.split("\n")
    # all_x = (w.count("-") for w in arr)
 #   max_depth = (max(list((w.count("-") for w in a.split("\n")))))
#print(num_of_leafs)
#print(num_of_nodes)
#print(max_depth)

#print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
#print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")