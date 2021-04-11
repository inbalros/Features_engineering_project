import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn import tree


#############################
#                           #
#  data sets - to evaluate  #
#                           #
#############################

#db_name = r'D:\phd\DB\Diabetes.csv'



def normalize_results(pred,index,dataset_name,method,number_of_classes,cluster_number,dataset_size,using_model):
    pred /= index
    pred_list = pred.tolist()
    pred_list = [dataset_name, method, number_of_classes, cluster_number, dataset_size, using_model] + pred_list
    return pred_list
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

def multi_class_DT_pred_old(train,test,data, x_names,y_names):
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
def predict_kfold_old(name,data, x_names,y_names,number_of_classes,paramK=5):
    kfold = KFold(paramK, True)
    index = 0
    size = data.shape[0]
    all_OVO_pred = 0
    all_multi_class_DT_pred =0
    for train, test in kfold.split(data):
        index += 1
        all_multi_class_DT_pred += multi_class_DT_pred(train,test,data, x_names,y_names)

    results_all_multi_class_DT_pred = normalize_results(all_multi_class_DT_pred,index,name,"decision_tree",number_of_classes,"-",size,"decision_tree")

    #dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred


##
def remove_duplicates(my_list):
  return list(dict.fromkeys(my_list))
def sort_f(f1,f2):
    f1 = str(f1).lower()
    f2 = str(f2).lower()
    if f2>f1:
        return f1+','+f2
    else:
        return f2+','+f1
def create_name(sort_stat,op,f1,f2=None):
    op_name = op.__name__
    if not sort_stat:
        if f2:
            return str(op_name)+'('+ sort_f(f1,f2) +')'
        else:
            return str(op_name)+'('+str(f1).lower()+')'
    else:
        if f2:
            return str(op_name)+'('+ str(f1).lower()+','+str(f2).lower() +')'
        else:
            return str(op_name)+'('+str(f1).lower()+')'
def prepare_new_feature(sort_stat,op, f1, f2):
    global dic_new_features
    name =create_name(sort_stat,op,f1,f2)
    if name not in dic_new_features:
        dic_new_features[name] = (op(f1,f2),f1,f2)
    return name

# unary , binary
def plus(f1,f2):
    return data[f1]+data[f2]
def minus(f1,f2):
    return data[f1]-data[f2]
def divide(f1,f2):
    return pd.Series([row[f1] / row[f2] if row[f2] != 0 else 0 for index, row in data.iterrows()])
def multiplication(f1,f2):
    return data[f1]*data[f2]

def create_new_features(features_unary,features_binary):
    global new_features_names
    global dic_new_features

    new_features_names = []
    for oper in operators_binary_direction_important:
        for f1 in features_binary:
            for f2 in features_binary:
                new_col=prepare_new_feature(True,oper,f1,f2)
                if len(dic_new_features[new_col][0].unique())>1 :
                    new_features_names.append(new_col)
    for oper in operators_binary_direction_NOT_important:
        for f1 in features_binary:
            for f2 in features_binary:
                new_col=prepare_new_feature(False,oper,f1,f2)
                if len(dic_new_features[new_col][0].unique())>1:
                    new_features_names.append(new_col)
    for oper in operators_unary:
        for f1 in features_unary:
            new_col =prepare_new_feature(False,oper,f1,None)
            if len(dic_new_features[new_col][0].unique())> 1 :
                new_features_names.append()


def multi_class_DT_pred(train,test,data, x_names,y_names):
    start_time = time.time()
    central_clf = DecisionTreeClassifier(max_depth=5).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
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
    #num_of_leaves = central_clf.tree_.n_leaves
    #precision, recall, fscore, support = metrics.precision_recall_fscore_support(
    #    pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    #return np.array(list((fit_time, pred_time,train_size,test_size, acu_test, acu_train, precision, recall, fscore)))
    #return (np.array(list((acu_test,num_of_leaves))),central_clf)
    return (np.array(list((acu_test)),central_clf))

def predict_kfold(data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,paramK=5,depth=None):
    kfold = KFold(paramK)
    index = 0
    size = data.shape[0]
    all_DT_pred =0
    for train, test in kfold.split(data):
        index += 1
        #arr,clf = multi_class_DT_pred(train,test,data, x_names,y_names)
        arr,clf = find_X_from_RF(train,test,data, x_names,y_names,criteria_Function,f_name,number_of_trees,depth)
        all_DT_pred += arr

    #results_all_multi_class_DT_pred = normalize_results(all_multi_class_DT_pred,index,name,"decision_tree",number_of_classes,"-",size,"decision_tree")
    all_DT_pred /= index
    pred_list = all_DT_pred.tolist()
    return pred_list,clf
    #dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred


def all_words_in_string(string):
    global added_f_names
    for word in added_f_names:
        if word not in string:
            return False
    return True


def find_X_from_RF(train,test,data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):
    start_time = time.time()
    if depth == None:
        central_clf = RandomForestClassifier(n_estimators=number_of_trees*3,random_state=None)\
            .fit(data.iloc[train][x_names],np.array(data.iloc[train][y_names]))
    else:
        central_clf = RandomForestClassifier(n_estimators=number_of_trees * 3,max_depth=depth,random_state=None)\
            .fit(data.iloc[train][x_names],np.array(data.iloc[train][y_names]))

    end_time = time.time()
    fit_time = end_time - start_time

    correct_trees = list()
    index_trees=0
    for cur_tree in central_clf.estimators_:
        if index_trees == number_of_trees:
            break
        if f_name:
            text_representation = tree.export_text(cur_tree, feature_names=x_names)
            if f_name in text_representation and all_words_in_string(text_representation):
                correct_trees.append(cur_tree)
                index_trees += 1
        else:
            correct_trees.append(cur_tree)
            index_trees += 1

    pred_acc=0
    criterion_trees=0
    for cur_tree in correct_trees:
        start_time = time.time()
        prediction = cur_tree.predict(data.iloc[test][x_names])  # the predictions labels
        end_time = time.time()
        pred_time = end_time - start_time
        acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
        pred_acc += acu_test
        criterion_trees += criteria_Function(cur_tree)

    if index_trees != number_of_trees:          ####check this####
        global criterion_base
        pred_acc =0
        criterion_trees = criterion_base
    else:
        pred_acc /= index_trees
        criterion_trees /= index_trees

    return (np.array(list((pred_acc, criterion_trees))), correct_trees)


def choose_best_feature(base_acu_test,base_criterion,criteria_Function,depth=None,number_of_kFolds=5,number_of_trees_per_fold=5,delete_used_f=False):
    best_critrion ={}
    global data
    global X_names

    for name in new_features_names:
        if name not in added_f_names:
            cur_data = data.copy()
            X_names_cur = X_names.copy()
            X_names_cur.append('new_f')
            cur_data.loc[:,'new_f']= dic_new_features[name][0]
            if delete_used_f:
                cur_data= cur_data.drop([dic_new_features[name][1]],axis=1)
                X_names_cur.remove(dic_new_features[name][1])
                if dic_new_features[name][1] != dic_new_features[name][2]:
                    cur_data = cur_data.drop([dic_new_features[name][2]],axis=1)
                    X_names_cur.remove(dic_new_features[name][2])
            new_pred,clf = predict_kfold(cur_data, X_names_cur, y_names,criteria_Function,'new_f',number_of_trees_per_fold,number_of_kFolds,depth)
            new_acu_test = new_pred[0]
            new_criterion = new_pred[1]
            if new_acu_test > (base_acu_test-0.03) and new_criterion < base_criterion:
                best_critrion[name] = (new_acu_test,new_criterion,clf)

    if len(best_critrion)==0:
        return None,(None,None,None)
    #min(value[1] for key,value in best_critrion.items())
    best_f = list(filter(None,[(key1,value1) if value1[1]==(min(value[1] for key,value in best_critrion.items())) else None for key1,value1 in best_critrion.items()]))[0]
    data.loc[:, best_f[0]] = dic_new_features[best_f[0]][0]
    X_names.append(best_f[0])
    if delete_used_f:
        data = data.drop([dic_new_features[best_f[0]][1]], axis=1)
        X_names.remove(dic_new_features[best_f[0]][1])
        if dic_new_features[best_f[0]][1] != dic_new_features[best_f[0]][2]:
            data = data.drop([dic_new_features[best_f[0]][2]], axis=1)
            X_names.remove(dic_new_features[best_f[0]][2])
    return best_f


def auto_F_E(number_of_kFolds,number_of_trees_per_fold,data_cur,X_names_cur,y_names_cur,features_unary,features_binary,depth,result_path,criteria_Function,delete_used_f=False):
    print(str(delete_used_f))
    global new_features_names
    global data
    global X_names
    global y_names
    global dic_new_features
    global added_f_names
    dic_new_features = {}
    new_features_names = []
    added_f_names = []
    data = data_cur
    X_names =X_names_cur.copy()
    print(X_names)
    y_names=y_names_cur
    base_pred,clf_list_base = \
        predict_kfold(data, X_names, y_names,criteria_Function,None,number_of_trees_per_fold,number_of_kFolds,depth)
    base_acu_test = base_pred[0]
    base_criterion = base_pred[1]

    global criterion_base
    criterion_base = base_criterion

    print(base_acu_test)
    print(base_criterion)
    acu_test = base_acu_test
    criterion_cur = base_criterion
    last_acu_test=base_acu_test
    f = open(result_path,"a")
    f.write("data-set: \n")
    f.write(str(delete_used_f)+ "\n")
    f.write(str(base_acu_test)+" \n")
    f.write(str(base_criterion)+" \n")
    rounds = 0
    last_criterion_cur=-1
    for num in range (5):
        create_new_features(features_unary,features_binary)
        new_features_names = list(dict.fromkeys(new_features_names))
        print(new_features_names)
        f.write("all new f: "+ str(new_features_names) + " \n")
        new_acc = max(base_acu_test,acu_test)  ######## what do you think about this?
        new_f_name,(acu_test,criterion_cur,clf) = choose_best_feature(new_acc,criterion_cur,criteria_Function,depth,number_of_kFolds,number_of_trees_per_fold,delete_used_f)
        print(X_names)
        f.write("X_names "+ str(X_names) + " \n")
        if new_f_name == None:
            break
        rounds=rounds+1
        added_f_names.append(new_f_name)
        if delete_used_f:  #do something in the future
            pass
        #features_binary.append(new_f_name)
        features_binary = X_names.copy()
        print(added_f_names)
        f.write("added: "+ str(added_f_names)+" \n")
        print(acu_test)
        print(criterion_cur)
        last_acu_test = acu_test
        last_criterion_cur = criterion_cur
        f.write(str(acu_test)+" \n")
        f.write(str(criterion_cur)+" \n")
    f.close()

    return base_acu_test,base_criterion,rounds,added_f_names,last_acu_test,last_criterion_cur

dic_new_features={}
new_features_names = []
added_f_names = []
data = None
X_names = None
y_names = None
criterion_base = 0

operators_binary_direction_important= [minus,divide]
# operators_binary_direction_important= []
operators_binary_direction_NOT_important= [multiplication,plus]
# operators_binary_direction_NOT_important= [plus]
operators_unary=[]

# for key, value in my_dict.items():

