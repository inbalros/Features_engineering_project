import numpy as np
import pandas as pd
from auto_by_criteria import *
# import auto_by_criteria as ac
# from automatic_FE.importances_features import *
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

from importances_features import *
import sys
import os


def plus(f1, f2):
    return f1 + f2

def minus(f1, f2):
    return f1 - f2

def divide(f1, f2):
    return pd.Series([f1_c / f2_c if f2_c != 0 else 0 for f1_c, f2_c in zip(f1, f2)])

def multiplication(f1, f2):
    return f1 * f2

def svc_binary_linear(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable =train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df, np.array(train_lable))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp['new'] = 1
        if i_u is not None:
            xp.loc[i_u, 'new'] = 0
    else:
        xp['new'] = 1

    return xp['new']

def svc_prediction_linear(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df, np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_linear(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]
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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train_fs_df, np.array(train_lable))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_
    distance_points = node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']

def svc_binary_poly(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df, np.array(train_lable))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp['new'] = 1
        if i_u is not None:
            xp.loc[i_u, 'new'] = 0
    else:
        xp['new'] = 1

    return xp['new']

def svc_prediction_poly(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df, np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_poly(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]
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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train_fs_df, np.array(train_lable))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_
    distance_points = node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']

def svc_binary_rbf(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df, np.array(train_lable))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp['new'] = 1
        if i_u is not None:
            xp.loc[i_u, 'new'] = 0
    else:
        xp['new'] = 1

    return xp['new']

def svc_prediction_rbf(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df, np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_rbf(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]
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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train_fs_df, np.array(train_lable))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_
    distance_points = node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']

def svc_binary_sigmoid(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df, np.array(train_lable))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp = pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True)
    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp['new'] = 1
        if i_u is not None:
            xp.loc[i_u, 'new'] = 0
    else:
        xp['new'] = 1

    return xp['new']

def svc_prediction_sigmoid(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]

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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df, np.array(train_lable))
    prediction_linear = cur_tree_linear.tree_._clf.predict(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    return pd.Series(prediction_linear)

def svc_distance_sigmoid(f1, f2):
    # var_name_f1 = [k for k, v in globals().items() if v is f1][0]
    # var_name_f2 = [k for k, v in globals().items() if v is f2][0]
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
        train_lable_l = [v for k, v in globals().items() if k == y_names]
        if len(train_lable_l) < 1:
            train_lable_l = [v for k, v in globals().items() if k is y_names]
        train_lable = train_lable_l[0]

    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train_fs_df, np.array(train_lable))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_
    distance_points = node._clf.decision_function(pd.DataFrame(data=[f1, f2]).T.copy().reset_index(drop=True))
    xp = pd.DataFrame(distance_points)
    xp['new'] = xp.apply(lambda row: np.linalg.norm(row), axis=1)
    return xp['new']


def find_X_from_RF_train_test(train, test, x_names, y_names, criteria_Function, f_name=None, number_of_trees=5,
                              depth=None):
    # arr_base = baseline_classifier_competition(train, test, data, x_names, y_names, criteria_Function, f_name,number_of_trees, depth)
    # handle_baseline_results(arr_base, data.copy(), depth)

    start_time = time.time()
    if ensemble == "RF":
        if depth == None:
            central_clf = RandomForestClassifier(n_estimators=number_of_trees, random_state=None) \
                .fit(train[x_names], np.array(train[y_names]))
        else:
            central_clf = RandomForestClassifier(n_estimators=number_of_trees, max_depth=depth, random_state=None) \
                .fit(train[x_names], np.array(train[y_names]))
    if ensemble == "XGB":
        if depth == None:
            central_clf = XGBClassifier(max_depth=100000, n_estimators=number_of_trees).fit(train[x_names],
                                                                                            np.array(train[y_names]))
        else:
            central_clf = XGBClassifier(max_depth=depth, n_estimators=number_of_trees).fit(train[x_names],
                                                                                           np.array(train[y_names]))
    if ensemble == "OB":
        if depth == None:
            central_clf = Stree(max_depth=100000).fit(train[x_names], np.array(train[y_names]))
        else:
            central_clf = Stree(max_depth=depth).fit(train[x_names], np.array(train[y_names]))

    end_time = time.time()
    fit_time = end_time - start_time

    criterion_trees = 0
    n_leaves_all = 0
    max_depth_all = 0
    node_count_all = 0
    size = 0

    # if index_trees != number_of_trees:          ####check this####
    #    global criterion_base
    #    pred_acc =0
    #    criterion_trees = criterion_base
    # else:
    start_time = time.time()
    prediction = central_clf.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    acu_test = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    pred_acc = acu_test
    # confusion_all += confusion_matrix(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)

    result_quality_cur = result_quality(y_true, y_pred)

    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    # only_one = 0
    # if len(un_true) == 1 or len(un_pred) == 1:
    #    print("bad " + str(x_names))
    #    only_one = 1
    # if len(un_true) == 1 and len(un_pred) == 1:
    # ("both " + str(x_names))
    #    y_true.append(0)
    #    y_true.append(1)
    #   y_pred.append(0)
    #    y_pred.append(1)
    #    y_true.append(0)
    #    y_true.append(1)
    #    y_pred.append(1)
    #    y_pred.append(0)
    # print("zero or ones")
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
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

    if ensemble == "RF":
        for cur_tree in central_clf.estimators_:
            criterion_trees += criteria_Function(cur_tree)
            n_leaves_all += cur_tree.tree_.n_leaves
            max_depth_all += cur_tree.tree_.max_depth
            node_count_all += cur_tree.tree_.node_count
            size += sys.getsizeof(cur_tree)
        n_leaves_all /= number_of_trees
        max_depth_all /= number_of_trees
        node_count_all /= number_of_trees
        criterion_trees /= number_of_trees
        size /= number_of_trees

    if ensemble == "XGB":
        booster = central_clf.get_booster()
        criterion_trees = criteria_Function(booster)
        n_leaves_all = criteria_number_of_leaves_xgboost(booster)
        max_depth_all = criteria_max_depth_xgboost(booster)
        node_count_all = criteria_number_of_nodes_xgboost(booster)
        size = get_xgboodt_size(booster)

    if ensemble == "OB":
        criterion_trees = criteria_Function(central_clf)
        n_leaves_all = criteria_number_of_leaves_oblique(central_clf)
        max_depth_all = criteria_max_depth_oblique(central_clf)
        node_count_all = criteria_number_of_nodes_oblique(central_clf)
        size = sys.getsizeof(central_clf)

    return np.array(list((result_quality_cur, criterion_trees, pred_acc, precision_all, recall_all, f_measure_all,
                          roc_all, prc_all, n_leaves_all,
                          max_depth_all, node_count_all, size))).tolist()


def from_name_to_column(recieve_data, name, number_of_f):
    # for i in range(1,number_of_f+1):
    #    locals()['att'+str(i)] = recieve_data['att'+str(i)]
    for col_name in recieve_data:
        globals()[col_name] = recieve_data[col_name]

    return eval(name)


def prepare_new_ds(data_testing, added_f_names, number_of_f):
    for name in added_f_names:
        if name not in data_testing:
            data_testing[name] = from_name_to_column(data_testing, name, number_of_f)

    return data_testing


dfAllPred = pd.DataFrame(
    columns=['dataset_name', 'number_of_features', 'number_of_classes', 'dataset_size', 'using_criterion',
             'using_result_quality',
             'result_quality_base', 'criteria_base', 'accuracy_base', 'precision_base', 'recall_base', 'f_measure_base',
             'roc_base', 'prc_base', 'n_leaves_base', 'max_depth_base', 'node_count_base',
             'number_of_folds', 'depth_max', 'number_of_trees_per_fold', 'number_of_rounds', 'delete_used_f',
             'added_features', 'all_features',
             'result_quality_after', 'criteria_after', 'accuracy_after', 'precision_after', 'recall_after',
             'f_measure_after', 'roc_after', 'prc_after', 'n_leaves_after', 'max_depth_after', 'node_count_after',
             'method','feature_family', 'train/test', 'size_model_before', 'size_db_train_before', 'size_db_test_before',
             'size_model_after', 'size_db_train_after', 'size_db_test_after'])

print(os.path.dirname(os.path.abspath(__file__)))
start_path = (os.path.dirname(os.path.abspath(__file__)))
one_back = os.path.dirname(start_path)

dataset_name = str(sys.argv[1])

dataset_path = str(sys.argv[2]) #r'Data\magic04\magic.csv'
f_number = int(str(sys.argv[3])) #10
y_number = int(f_number)+1


delete_used_choice = eval(sys.argv[4])
ensemble = str(sys.argv[5])  # RF, XGB , OB
eval_result = str(sys.argv[6])  # acc, F, PRC
criteria_choice = str(sys.argv[7])  # leaves, depth,nodes

feature_family= str(sys.argv[8])  # math, svc, all

result_path = os.path.join(one_back, r"results_1\results_" + dataset_name + ".txt")

# result_path = r"..\results\results_"+dataset_name+".txt"

index1 = 1


def write_to_excel_dfAllPred(final=False):
    global index1
    if final:
        writerResults = pd.ExcelWriter(os.path.join(one_back, r"results_1\results_" + dataset_name + "_" + str(
            ensemble) + "_" + str(delete_used_choice) + "_" + str(eval_result) + "_" + str(
            criteria_choice) + "_" + str(feature_family) +"_final.xlsx"))
    else:
        writerResults = pd.ExcelWriter(os.path.join(one_back, r"results_1\results_" + dataset_name + "_" + str(
            ensemble) + "_" + str(delete_used_choice) + "_" + str(eval_result) + "_" + str(criteria_choice) + "_" + str(feature_family) + "_" + str(
            index1) + ".xlsx"))

    index1 += 1
    dfAllPred.to_excel(writerResults, 'results')
    writerResults.save()


###############################
#                             #
#      all the criterion      #
#         functions           #
#                             #
###############################

def criteria_number_of_leaves_RF(tree):
    return tree.tree_.n_leaves


def criteria_max_depth_RF(tree):
    return tree.tree_.max_depth


def criteria_number_of_nodes_RF(tree):
    return tree.tree_.node_count


def get_xgboodt_size(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    return sys.getsizeof(a)


def criteria_number_of_leaves_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    number_of_leafs = 0
    for tree in a:
        number_of_leafs += tree.count("leaf")
    number_of_leafs /= len(a)
    return number_of_leafs


def criteria_number_of_nodes_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    num_of_nodes = 0
    for tree in a:
        num_of_nodes += tree.count("nodeid")
    num_of_nodes /= len(a)
    return num_of_nodes


def criteria_max_depth_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    max_depth_all = 0
    for tree in a:
        all_lines = tree.split('\n')
        max_depth = -1
        regex = re.compile(r"(?<=\b:)\w+")
        for line in all_lines:
            match = re.search('"depth": (\d+)', line)
            if match:
                mat = int(match.group(1))
                if max_depth < mat:
                    max_depth = mat
        # we need to add one for the depth of the leaf
        max_depth += 1
        max_depth_all += max_depth
    max_depth_all /= len(a)
    return max_depth_all


def criteria_number_of_leaves_oblique(clf):
    a = str(clf)
    num_of_leafs = (a.count("Leaf"))
    return num_of_leafs


def criteria_number_of_nodes_oblique(clf):
    a = str(clf)
    num_of_nodes = (a.count("\n"))
    return num_of_nodes


def criteria_max_depth_oblique(clf):
    a = str(clf)
    max_depth = (max(list((w.count("-") for w in a.split("\n")))))
    return max_depth


###############################
#                             #
#    all the result_quality   #
#         functions           #
#                             #
###############################

def result_quality_accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def result_quality_F(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='weighted')


def result_quality_PRC(y_true, y_pred):
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1
    return prc_all


###############################
#                             #
#      start the data-sets    #
#         experiments         #
#                             #
###############################

if eval_result == "acc":
    result_quality = result_quality_accuracy
elif eval_result == "F":
    result_quality = result_quality_F
elif eval_result == "PRC":
    result_quality = result_quality_PRC

number_of_kFolds = 3
number_of_trees_per_fold = 1000
depth = None  ###################################### add as parameter
all_criterions_RF = {'max_depth': criteria_max_depth_RF, 'number_of_leaves': criteria_number_of_leaves_RF,
                     'number_of_nodes': criteria_number_of_nodes_RF}
all_criterions_XGB = {'max_depth': criteria_max_depth_xgboost, 'number_of_leaves': criteria_number_of_leaves_xgboost,
                      'number_of_nodes': criteria_number_of_nodes_xgboost}
all_criterions_OB = {'max_depth': criteria_max_depth_oblique, 'number_of_leaves': criteria_number_of_leaves_oblique,
                     'number_of_nodes': criteria_number_of_nodes_oblique}
# all_criterions={'number_of_leaves':criteria_number_of_leaves,'number_of_nodes':criteria_number_of_nodes}

if ensemble == "RF":
    if criteria_choice == "leaves":  # leaves, depth,nodes
        all_criterions = {'number_of_leaves': criteria_number_of_leaves_RF}
    elif criteria_choice == "depth":  # leaves, depth,nodes
        all_criterions = {'max_depth': criteria_max_depth_RF}
    elif criteria_choice == "nodes":  # leaves, depth,nodes
        all_criterions = {'number_of_nodes': criteria_number_of_nodes_RF}

elif ensemble == "XGB":
    if criteria_choice == "leaves":  # leaves, depth,nodes
        all_criterions = {'number_of_leaves': criteria_number_of_leaves_xgboost}
    elif criteria_choice == "depth":  # leaves, depth,nodes
        all_criterions = {'max_depth': criteria_max_depth_xgboost}
    elif criteria_choice == "nodes":  # leaves, depth,nodes
        all_criterions = {'number_of_nodes': criteria_number_of_nodes_xgboost}

elif ensemble == "OB":
    if criteria_choice == "leaves":  # leaves, depth,nodes
        all_criterions = {'number_of_leaves': criteria_number_of_leaves_oblique}
    elif criteria_choice == "depth":  # leaves, depth,nodes
        all_criterions = {'max_depth': criteria_max_depth_oblique}
    elif criteria_choice == "nodes":  # leaves, depth,nodes
        all_criterions = {'number_of_nodes': criteria_number_of_nodes_oblique}


def encode_categorical(df, label):
    df = df.replace(['?'], None)
    try_data = df.copy().select_dtypes(include='object')
    category_columns = try_data.columns

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(mode(df[col].astype(str)).mode[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    le = LabelEncoder()
    for i in category_columns:
        df[i] = le.fit_transform(df[i].astype(str))
        if i is not label:
            df[i] += 1

    return df


print(dataset_name)
data_path = os.path.join(one_back, dataset_path)
choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, y_number+1)],skiprows = 1)
X_names_ds = ["att" + str(i) for i in range(1, f_number+1)]
y_names = "att"+str(y_number)
choosen_data = encode_categorical(choosen_data,y_names)
un_class = len(choosen_data[y_names].unique())
print(un_class)
db_name = dataset_name
data_chosen = choosen_data[X_names_ds].copy()
features_unary = []
features_binary = ["att" + str(i) for i in range(1, f_number+1)]
label = "att"+str(y_number)
data_chosen[label] = choosen_data[label]


test_train_split = True
NEW_data_testing = None


for ind in range(1,6):

    index1 = 1
    dfAllPred = pd.DataFrame(
        columns=['dataset_name', 'number_of_features', 'number_of_classes', 'dataset_size', 'using_criterion',
                 'using_result_quality',
                 'result_quality_base', 'criteria_base', 'accuracy_base', 'precision_base', 'recall_base',
                 'f_measure_base',
                 'roc_base', 'prc_base', 'n_leaves_base', 'max_depth_base', 'node_count_base',
                 'number_of_folds', 'depth_max', 'number_of_trees_per_fold', 'number_of_rounds', 'delete_used_f',
                 'added_features', 'all_features',
                 'result_quality_after', 'criteria_after', 'accuracy_after', 'precision_after', 'recall_after',
                 'f_measure_after', 'roc_after', 'prc_after', 'n_leaves_after', 'max_depth_after', 'node_count_after',
                 'method','feature_family','train/test', 'size_model_before', 'size_db_train_before', 'size_db_test_before',
                 'size_model_after', 'size_db_train_after', 'size_db_test_after'])


    data_chosen = choosen_data.copy().sample(frac=1)
    data_chosen = data_chosen.reset_index(drop=True)
    data_chosen = data_chosen.dropna().reset_index(drop=True)

    #print(data_chosen)

    if test_train_split == True:
        kfold = KFold(4)
        index = 0
        size = data_chosen.shape[0]
        all_DT_pred = 0
        for train, test in kfold.split(data_chosen):
            NEW_data_chosen = data_chosen.iloc[train]
            #print(NEW_data_chosen)
            NEW_data_chosen = NEW_data_chosen.reset_index(drop=True)
            #print(NEW_data_chosen)
            NEW_data_testing = data_chosen.iloc[test]
            NEW_data_testing = NEW_data_testing.reset_index(drop=True)
            print(len(train))
            print(len(test))
            #print(NEW_data_chosen)
            break
        data_chosen = NEW_data_chosen

    # c1 = from_name_to_column(data_chosen,'minus(att3,att1)',3)
    # c2 = from_name_to_column(data_chosen,'minus(divide(att3,att2),multiplication(att2,divide(att2,minus(att3,att1))))',3)

    size = sys.getsizeof(data_chosen)

    # for delete_used_f in [True,False]:
    for delete_used_f in [delete_used_choice]:
        # for delete_used_f in [False]:
        for criterion_name, criterion_function in all_criterions.items():
            # for cur_depth in [None,1,2,3,5,10,15,20,25]:
            for cur_depth in [None]:
                set_params(str(f_number), str(un_class), criterion_name, str(number_of_kFolds),
                           str(number_of_trees_per_fold), str(delete_used_f))
                before_quality_result, before_criterion, before_acu_test, rounds, added_f_names, after_quality_result, criterion_after, acu_test, precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after, last_x_name = \
                    auto_F_E(result_quality, ensemble, number_of_kFolds, number_of_trees_per_fold, data_chosen.copy(),
                             X_names_ds, y_names, features_unary, features_binary, cur_depth, result_path,
                             criterion_function, delete_used_f,feature_family)

                dfAllPred.loc[len(dfAllPred)] = np.array(
                    [db_name, str(f_number), str(un_class), str(data_chosen.shape[0]), criterion_name, eval_result,
                     str(before_quality_result), before_criterion, str(before_acu_test),
                     str(precision_before), str(recall_before), str(f_measure_before), str(roc_before), str(prc_before),
                     str(n_leaves_before), str(max_depth_before), str(node_count_before),
                     str(number_of_kFolds), str(cur_depth), str(number_of_trees_per_fold), str(rounds), str(delete_used_f),
                     str(added_f_names),
                     str(last_x_name), str(after_quality_result), str(criterion_after), str(acu_test),
                     str(precision_after), str(recall_after), str(f_measure_after),
                     str(roc_after), str(prc_after), str(n_leaves_after),
                     str(max_depth_after), str(node_count_after), "ours - " + str(ensemble),str(feature_family), "train", "", "", "", "", "",
                     ""])
                write_to_excel_dfAllPred()

                '''
                before_quality_result, before_criterion, before_acu_test, after_quality_result, criterion_after, acu_test, precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, \
                precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after = get_data_for_EXCEL()
    
                dfAllPred.loc[len(dfAllPred)] = np.array(
                    [db_name, str(f_number), str(un_class), str(data_chosen.shape[0]), criterion_name, eval_result,
                     str(before_quality_result),
                     before_criterion, str(before_acu_test),
                     str(precision_before), str(recall_before), str(f_measure_before), str(roc_before), str(prc_before),
                     str(n_leaves_before), str(max_depth_before), str(node_count_before),
                     str(number_of_kFolds), str(cur_depth), str(number_of_trees_per_fold), str(rounds), str(delete_used_f),
                     str(added_f_names),
                     str(last_x_name), str(after_quality_result), str(criterion_after), str(acu_test),
                     str(precision_after), str(recall_after), str(f_measure_after),
                     str(roc_after), str(prc_after), str(n_leaves_after),
                     str(max_depth_after), str(node_count_after), "oblique", "train", "", "", "", "", "", ""])
                write_to_excel_dfAllPred()
                '''
                # def find_X_from_RF_train_test(train, test, x_names, y_names, criteria_Function, f_name=None,
                #                              number_of_trees=5, depth=None)
                # predict_kfold(data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,paramK=5,depth=None)
                if test_train_split == True:
                    (before_quality_result, r_test_criterion_base, r_test_acu_base, r_test_precision_base,
                     r_test_recall_base, r_test_f_measure_base, r_test_roc_base, r_test_prc_base,
                     r_test_n_leaves_base, r_test_max_depth_base, r_test_node_count_base, central_clf_size_before) = \
                        find_X_from_RF_train_test(data_chosen, NEW_data_testing, X_names_ds, y_names, criterion_function,
                                                  None, number_of_trees_per_fold, cur_depth)

                    '''
                    arr_res = find_X_from_baseline_STree_train_test(data_chosen, NEW_data_testing, X_names_ds, y_names,
                                                                    criterion_function, None, number_of_trees_per_fold,
                                                                    cur_depth)
    
                    handle_baseline_results_STree(arr_res, NEW_data_testing, cur_depth, X_names_ds, "test_all_f", "before")
                    
                    arr_res = find_X_from_baseline_xgboost_train_test(data_chosen, NEW_data_testing, X_names_ds, y_names,
                                                                    criterion_function, None, number_of_trees_per_fold,
                                                                    cur_depth)
                    handle_baseline_results_xgboost(arr_res, NEW_data_testing, cur_depth, X_names_ds, "test_all_f")
                    '''
                    if rounds > 0:

                        data_chosen_new = pd.DataFrame()
                        data_chosen_new = prepare_new_ds(data_chosen.copy(), added_f_names, f_number)
                        NEW_data_testing_after = prepare_new_ds(NEW_data_testing.copy(), added_f_names, f_number)

                        (after_quality_result, r_test_criterion, r_test_acu, r_test_precision, r_test_recall, r_test_f_measure,
                         r_test_roc, r_test_prc,
                         r_test_n_leaves, r_test_max_depth, r_test_node_count, central_clf_size_after) = \
                            find_X_from_RF_train_test(data_chosen_new, NEW_data_testing_after, last_x_name, y_names,
                                                      criterion_function, None, number_of_trees_per_fold, cur_depth)

                        dfAllPred.loc[len(dfAllPred)] = np.array(
                            [db_name, str(f_number), str(un_class), str(NEW_data_testing_after.shape[0]), criterion_name,
                             eval_result, str(before_quality_result),
                             r_test_criterion_base, str(r_test_acu_base),
                             str(r_test_precision_base), str(r_test_recall_base), str(r_test_f_measure_base),
                             str(r_test_roc_base), str(r_test_prc_base),
                             str(r_test_n_leaves_base), str(r_test_max_depth_base), str(r_test_node_count_base),
                             str(number_of_kFolds), str(cur_depth), str(number_of_trees_per_fold), str(rounds),
                             str(delete_used_f),
                             str(added_f_names),
                             str(last_x_name), str(after_quality_result), str(r_test_criterion), str(r_test_acu),
                             str(r_test_precision), str(r_test_recall), str(r_test_f_measure),
                             str(r_test_roc), str(r_test_prc), str(r_test_n_leaves),
                             str(r_test_max_depth), str(r_test_node_count), "ours - " + str(ensemble),str(feature_family),"test",
                             str(central_clf_size_before), str(sys.getsizeof(data_chosen)),
                             str(sys.getsizeof(NEW_data_testing)), str(central_clf_size_after),
                             str(sys.getsizeof(data_chosen_new[last_x_name])),
                             str(sys.getsizeof(NEW_data_testing_after[last_x_name]))])

                        write_to_excel_dfAllPred()

                    '''
                    arr_res = find_X_from_baseline_STree_train_test(data_chosen_new, NEW_data_testing_after, last_x_name,
                                                                    y_names, criterion_function, None,
                                                                    number_of_trees_per_fold, cur_depth)
                    handle_baseline_results_STree(arr_res, NEW_data_testing_after, cur_depth, last_x_name, "test_new_f",
                                                  "after")
    
                    before_quality_result, before_criterion, before_acu_test, after_quality_result, criterion_after, acu_test, precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, \
                    precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after = get_data_for_EXCEL()
    
                    dfAllPred.loc[len(dfAllPred)] = np.array(
                        [db_name, str(f_number), str(un_class), str(NEW_data_testing_after.shape[0]), criterion_name,
                         eval_result,
                         str(before_quality_result),
                         before_criterion, str(before_acu_test),
                         str(precision_before), str(recall_before), str(f_measure_before), str(roc_before), str(prc_before),
                         str(n_leaves_before), str(max_depth_before), str(node_count_before),
                         str(number_of_kFolds), str(cur_depth), str(number_of_trees_per_fold), str(rounds),
                         str(delete_used_f),
                         str(added_f_names),
                         str(last_x_name), str(after_quality_result), str(criterion_after), str(acu_test),
                         str(precision_after), str(recall_after), str(f_measure_after),
                         str(roc_after), str(prc_after), str(n_leaves_after),
                         str(max_depth_after), str(node_count_after), "oblique", "test", "", "", "", "", "", ""])
                    write_to_excel_dfAllPred()
    
                    
                    arr_res = find_X_from_baseline_xgboost_train_test(data_chosen_new, NEW_data_testing, last_x_name, y_names,
                                                                      criterion_function, None, number_of_trees_per_fold,
                                                                      cur_depth)
                    handle_baseline_results_xgboost(arr_res, NEW_data_testing, cur_depth, last_x_name, "test_new_f")
                    
    
                if rounds > 0:
                    importance_experiment(ensemble, eval_result, db_name, criterion_name, delete_used_f,
                                          data_chosen, data_chosen_new, added_f_names, last_x_name, X_names_ds, y_names,
                                          1000)
                   '''
    if rounds > 0:
        break

# importance_experiment();
write_to_excel_dfAllPred(True)
