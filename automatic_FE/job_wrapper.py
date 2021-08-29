import numpy as np
import pandas as pd
from auto_by_criteria import *
#import auto_by_criteria as ac
#from automatic_FE.importances_features import *
from importances_features import *

import sys
import os


def plus(f1,f2):
    return f1+f2
def minus(f1,f2):
    return f1-f2
def divide(f1,f2):
    return pd.Series([f1_c / f2_c if f2_c != 0 else 0 for f1_c, f2_c in zip(f1,f2)])
def multiplication(f1,f2):
    return f1*f2

def svc_binary_linear(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new :
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable =  data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable=[v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df,np.array(train_lable))
    node= cur_tree_linear.tree_
    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = cur_tree_linear.splitter_.part(indices)
    xp['new'] = 1
    if i_u is not None:
        xp.loc[i_u,'new'] = 0
    return xp['new']

def svc_prediction_linear(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df,np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_linear(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]
    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df,np.array(train_lable))
    #prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node= cur_tree_linear.tree_
    distance_points =  node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']


def svc_binary_poly(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new :
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable =  data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable=[v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df,np.array(train_lable))
    node= cur_tree_linear.tree_
    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = cur_tree_linear.splitter_.part(indices)
    xp['new'] = 1
    if i_u is not None:
        xp.loc[i_u,'new'] = 0
    return xp['new']

def svc_prediction_poly(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df,np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_poly(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]
    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df,np.array(train_lable))
    #prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node= cur_tree_linear.tree_
    distance_points =  node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']

def svc_binary_rbf(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new :
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable =  data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable=[v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df,np.array(train_lable))
    node= cur_tree_linear.tree_
    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = cur_tree_linear.splitter_.part(indices)
    xp['new'] = 1
    if i_u is not None:
        xp.loc[i_u,'new'] = 0
    return xp['new']

def svc_prediction_rbf(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df,np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_rbf(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]
    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df,np.array(train_lable))
    #prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node= cur_tree_linear.tree_
    distance_points =  node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']

def svc_binary_sigmoid(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new :
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable =  data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable=[v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df,np.array(train_lable))
    node= cur_tree_linear.tree_
    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = cur_tree_linear.splitter_.part(indices)
    xp['new'] = 1
    if i_u is not None:
        xp.loc[i_u,'new'] = 0
    return xp['new']

def svc_prediction_sigmoid(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df,np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_sigmoid(f1,f2):
    #var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    #var_name_f2 = [k for k, v in globals().items() if v is f2][0]
    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f1):
                var_name_f1 = k
                break

    for k, v in globals().items():
        if type(v) == pd.core.series.Series:
            if list(v) == list(f2):
                var_name_f2 = k
                break

    if var_name_f1 in data_chosen_new and var_name_f2 in data_chosen_new:
        train_fs_df = pd.DataFrame(data=[data_chosen_new[var_name_f1], data_chosen_new[var_name_f2]]).T
        train_lable = data_chosen_new[y_names]
    else:
        train_fs_df = pd.DataFrame(data=[f1, f2]).T
        train_lable = [v for k, v in globals().items() if k is y_names][0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df,np.array(train_lable))
    #prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node= cur_tree_linear.tree_
    distance_points =  node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']





operators_binary_direction_important= [minus,divide]
#operators_binary_direction_important= []
operators_binary_direction_NOT_important= [multiplication,plus]




def find_X_from_RF_train_test(train,test,x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):
    #arr_base = baseline_classifier_competition(train, test, data, x_names, y_names, criteria_Function, f_name,number_of_trees, depth)
    #handle_baseline_results(arr_base, data.copy(), depth)

    start_time = time.time()
    if depth == None:
        central_clf = RandomForestClassifier(n_estimators=number_of_trees * 3,random_state=None)\
            .fit(train[x_names],np.array(train[y_names]))
    else:
        central_clf = RandomForestClassifier(n_estimators=number_of_trees * 3,max_depth=depth,random_state=None)\
            .fit(train[x_names],np.array(train[y_names]))


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
    precision_all=0
    recall_all=0
    f_measure_all=0
    roc_all=0
    prc_all=0
    criterion_trees=0
    #confusion_all=0

    n_leaves_all =0
    max_depth_all=0
    node_count_all=0

    if index_trees != number_of_trees:          ####check this####
        global criterion_base
        pred_acc =0
        criterion_trees = criterion_base
    else:
        for cur_tree in correct_trees:
            start_time = time.time()
            prediction = cur_tree.predict(test[x_names])  # the predictions labels
            end_time = time.time()
            pred_time = end_time - start_time
            acu_test = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
            pred_acc += acu_test
            #confusion_all += confusion_matrix(pd.Series.tolist(data.iloc[test][y_names]), prediction)
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
                #print("zero or ones")
            criterion_trees += criteria_Function(cur_tree)
            precision_all += metrics.precision_score(y_true, y_pred,average='weighted')
            recall_all += metrics.recall_score(y_true, y_pred,average='weighted')
            f_measure_all += metrics.f1_score(y_true, y_pred,average='weighted')
            try:
                roc_all += metrics.roc_auc_score(y_true, y_pred)
            except:
                #print("exception_roc")
                roc_all +=  0
            try:
                precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
                prc_all += metrics.auc(recall_prc, precision_prc)
            except:
                #print("exception_prc - N")
                prc_all += 1

            ############## All the criterion options on cur_tree:
            n_leaves_all += cur_tree.tree_.n_leaves
            max_depth_all += cur_tree.tree_.max_depth
            node_count_all += cur_tree.tree_.node_count


        pred_acc /= index_trees
        #confusion_all = confusion_all/index_trees
        precision_all/= index_trees
        recall_all/= index_trees
        f_measure_all/= index_trees
        roc_all/= index_trees
        prc_all/= index_trees
        n_leaves_all /= index_trees
        max_depth_all/= index_trees
        node_count_all/= index_trees
        criterion_trees /= index_trees

    return np.array(list((pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))).tolist()


# def from_name_to_column(recieve_data,name):
#     name= name.translate({ord(i): None for i in '()'})
#     first_col = name.split(',', 1)[0]
#     secon_col = name.split(',', 1)[1]
#     selected_oper = None
#     selected_oper_name=None
#     for oper in operators_binary_direction_important:
#         if str(oper.__name__) in first_col:
#             selected_oper = oper
#             selected_oper_name = str(oper.__name__)
#             break
#     if selected_oper is None:
#         for oper in operators_binary_direction_NOT_important:
#             if str(oper.__name__) in first_col:
#                 selected_oper = oper
#                 selected_oper_name = str(oper.__name__)
#                 break
#     first_col = first_col.replace(selected_oper_name, '')
#     if first_col in recieve_data:
#         f1=recieve_data[first_col]
#     else:
#         f1 = from_name_to_column(recieve_data,first_col)
#     if secon_col in recieve_data:
#         f2 = recieve_data[secon_col]
#     else:
#         f2 = from_name_to_column(recieve_data, secon_col)

    # return oper(f1,f2)

def from_name_to_column(recieve_data,name,number_of_f):
    #for i in range(1,number_of_f+1):
    #    locals()['att'+str(i)] = recieve_data['att'+str(i)]
    for col_name in recieve_data:
        globals()[col_name] = recieve_data[col_name]

    return eval(name)


def prepare_new_ds(data_testing,added_f_names,number_of_f):
    for name in added_f_names:
        if name not in data_testing:
            data_testing[name]= from_name_to_column(data_testing,name,number_of_f)

    return data_testing


dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
         'number_of_folds','depth_max','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features','all_features',
         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after','method'])


print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)

dataset_name= str(sys.argv[1])

result_path=os.path.join(one_back,  r"results_1\results_"+dataset_name+".txt")


#result_path = r"..\results\results_"+dataset_name+".txt"

index1 = 1
def write_to_excel_dfAllPred():
    global index1
    writerResults = pd.ExcelWriter(os.path.join(one_back,  r"results_1\results_all_"+dataset_name+"_"+str(index1)+".xlsx"))
    index1+=1
    dfAllPred.to_excel(writerResults,'results')
    writerResults.save()


###############################
#                             #
#      all the criterion      #
#         functions           #
#                             #
###############################

def criteria_number_of_leaves(tree):
    return tree.tree_.n_leaves

def criteria_max_depth(tree):
    return tree.tree_.max_depth

def criteria_number_of_nodes(tree):
    return tree.tree_.node_count

###############################
#                             #
#      start the data-sets    #
#         experiments         #
#                             #
###############################

number_of_kFolds = 10
number_of_trees_per_fold = 10
depth = None  ###################################### add as parameter
all_criterions={'max_depth': criteria_max_depth ,'number_of_leaves':criteria_number_of_leaves,'number_of_nodes':criteria_number_of_nodes}
#all_criterions={'number_of_leaves':criteria_number_of_leaves,'number_of_nodes':criteria_number_of_nodes}


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



test_train_split=True
NEW_data_testing=None

if test_train_split == True:
    kfold = KFold(7)
    index = 0
    size = data_chosen.shape[0]
    all_DT_pred = 0
    for train, test in kfold.split(data_chosen):
        NEW_data_chosen = data_chosen.iloc[train]
        print(NEW_data_chosen)
        NEW_data_chosen = NEW_data_chosen.reset_index(drop=True)
        print(NEW_data_chosen)
        NEW_data_testing = data_chosen.iloc[test]
        NEW_data_testing = NEW_data_testing.reset_index(drop=True)
        print(len(train))
        print(len(test))
        break
    data_chosen = NEW_data_chosen

#c1 = from_name_to_column(data_chosen,'minus(att3,att1)',3)
#c2 = from_name_to_column(data_chosen,'minus(divide(att3,att2),multiplication(att2,divide(att2,minus(att3,att1))))',3)

for delete_used_f in [True,False]:
#for delete_used_f in [False]:
    for criterion_name, criterion_function in all_criterions.items():
        #for cur_depth in [None,1,2,3,5,10,15,20,25]:
        for cur_depth in [None]:
            set_params(str(f_number),str(un_class), criterion_name, str(number_of_kFolds),str(number_of_trees_per_fold), str(delete_used_f))
            before_acu_test, before_criterion, rounds, added_f_names, acu_test, criterion_after ,precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after, last_x_name, new_ds =\
                auto_F_E(number_of_kFolds,number_of_trees_per_fold,data_chosen.copy(),X_names_ds,y_names,features_unary,features_binary,cur_depth,result_path,criterion_function,delete_used_f)

            dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data_chosen.shape[0]),criterion_name,str(before_acu_test), before_criterion,
                                                      str(precision_before),str(recall_before),str(f_measure_before),str(roc_before),str(prc_before),str(n_leaves_before),str(max_depth_before),str(node_count_before),
                                                      str(number_of_kFolds),str(cur_depth),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),
                                                      str(last_x_name),str(acu_test),str(criterion_after),
                                                      str(precision_after), str(recall_after), str(f_measure_after),
                                                      str(roc_after), str(prc_after), str(n_leaves_after),
                                                      str(max_depth_after), str(node_count_after),"ours"])
            write_to_excel_dfAllPred()


            #def find_X_from_RF_train_test(train, test, x_names, y_names, criteria_Function, f_name=None,
            #                              number_of_trees=5, depth=None)
            #predict_kfold(data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,paramK=5,depth=None)
            if test_train_split == True:
                (r_test_acu_base, r_test_criterion_base, r_test_precision_base, r_test_recall_base, r_test_f_measure_base, r_test_roc_base, r_test_prc_base,
                 r_test_n_leaves_base, r_test_max_depth_base, r_test_node_count_base) = \
                    find_X_from_RF_train_test(data_chosen,NEW_data_testing, X_names_ds, y_names, criterion_function, None, number_of_trees_per_fold,cur_depth)

                arr_res = find_X_from_baseline_STree_train_test(data_chosen, NEW_data_testing, X_names_ds, y_names,
                                                                criterion_function, None, number_of_trees_per_fold,
                                                                cur_depth)
                
                handle_baseline_results_STree(arr_res, NEW_data_testing, cur_depth, X_names_ds, "test_all_f")
                '''
                arr_res = find_X_from_baseline_xgboost_train_test(data_chosen, NEW_data_testing, X_names_ds, y_names,
                                                                criterion_function, None, number_of_trees_per_fold,
                                                                cur_depth)
                handle_baseline_results_xgboost(arr_res, NEW_data_testing, cur_depth, X_names_ds, "test_all_f")
                '''

                data_chosen_new = pd.DataFrame()
                data_chosen_new = prepare_new_ds(data_chosen.copy(),added_f_names,f_number)
                NEW_data_testing = prepare_new_ds(NEW_data_testing,added_f_names,f_number)

                (r_test_acu, r_test_criterion, r_test_precision, r_test_recall, r_test_f_measure, r_test_roc, r_test_prc,
                 r_test_n_leaves, r_test_max_depth, r_test_node_count) = \
                    find_X_from_RF_train_test(data_chosen_new,NEW_data_testing, last_x_name, y_names, criterion_function, None, number_of_trees_per_fold,cur_depth)

                dfAllPred.loc[len(dfAllPred)] = np.array(
                    [db_name, str(f_number), str(un_class), str(NEW_data_testing.shape[0]), criterion_name+"_testing", str(r_test_acu_base),
                     r_test_criterion_base,
                     str(r_test_precision_base), str(r_test_recall_base), str(r_test_f_measure_base), str(r_test_roc_base), str(r_test_prc_base),
                     str(r_test_n_leaves_base), str(r_test_max_depth_base), str(r_test_node_count_base),
                     str(number_of_kFolds), str(cur_depth), str(number_of_trees_per_fold), str(rounds), str(delete_used_f),
                     str(added_f_names),
                     str(last_x_name), str(r_test_acu), str(r_test_criterion),
                     str(r_test_precision), str(r_test_recall), str(r_test_f_measure),
                     str(r_test_roc), str(r_test_prc), str(r_test_n_leaves),
                     str(r_test_max_depth), str(r_test_node_count),"ours"])

                write_to_excel_dfAllPred()


                arr_res = find_X_from_baseline_STree_train_test(data_chosen_new,NEW_data_testing, last_x_name, y_names, criterion_function, None, number_of_trees_per_fold,cur_depth)
                handle_baseline_results_STree(arr_res, NEW_data_testing, cur_depth, last_x_name, "test_new_f")
                '''
                arr_res = find_X_from_baseline_xgboost_train_test(data_chosen_new, NEW_data_testing, last_x_name, y_names,
                                                                  criterion_function, None, number_of_trees_per_fold,
                                                                  cur_depth)
                handle_baseline_results_xgboost(arr_res, NEW_data_testing, cur_depth, last_x_name, "test_new_f")
                '''
            #if rounds>0:
                #importance_experiment(db_name,f_number,un_class,criterion_name,delete_used_f,data_chosen,new_ds,added_f_names,last_x_name,X_names_ds,y_names,100,criterion_function)


