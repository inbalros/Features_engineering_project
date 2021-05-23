import numpy as np
import pandas as pd
from automatic_FE.auto_by_criteria import *
from automatic_FE.importances_features import *

import sys
import os

dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
         'number_of_folds','depth_max','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features','all_features',
         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after'])


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



for delete_used_f in [True,False]:
    for criterion_name, criterion_function in all_criterions.items():
        for cur_depth in [None,1,2,3,5,10,15,20,25]:
            before_acu_test, before_criterion, rounds, added_f_names, acu_test, criterion_after ,precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after, last_x_name, new_ds =\
                auto_F_E(number_of_kFolds,number_of_trees_per_fold,data_chosen.copy(),X_names_ds,y_names,features_unary,features_binary,cur_depth,result_path,criterion_function,delete_used_f)

            dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data_chosen.shape[0]),criterion_name,str(before_acu_test), before_criterion,
                                                      str(precision_before),str(recall_before),str(f_measure_before),str(roc_before),str(prc_before),str(n_leaves_before),str(max_depth_before),str(node_count_before),
                                                      str(number_of_kFolds),str(cur_depth),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),
                                                      str(last_x_name),str(acu_test),str(criterion_after),
                                                      str(precision_after), str(recall_after), str(f_measure_after),
                                                      str(roc_after), str(prc_after), str(n_leaves_after),
                                                      str(max_depth_after), str(node_count_after)])
            write_to_excel_dfAllPred()

            if rounds>0:
                importance_experiment(db_name,f_number,un_class,criterion_name,delete_used_f,data_chosen,new_ds,added_f_names,last_x_name,X_names_ds,y_names,100,criterion_function)











