import numpy as np
import pandas as pd
from automatic_FE.auto_by_criteria import *
import sys

dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'Accuracy_base', 'Criteria_base','number_of_folds','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features',
         'accuracy_after', 'criteria_after'])

dataset_name= str(sys.argv[1])
result_path = r"..\results\results_"+dataset_name+".txt"

index = 1
def write_to_excel():
    global index
    writerResults = pd.ExcelWriter(r"..\results\results_"+dataset_name+"_"+str(index)+".xlsx")
    index+=1
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

number_of_kFolds = 5
number_of_trees_per_fold = 5
depth = None  ###################################### add as parameter
all_criterions={'max_depth': criteria_max_depth ,'number_of_leaves':criteria_number_of_leaves,'number_of_nodes':criteria_number_of_nodes}


if(dataset_name=='magic'):
    print(dataset_name)
    data_path = r'..\Data\magic04\magic.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 12)])
    X_names_ds = ["att" + str(i) for i in range(1, 11)]
    y_names = "att11"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'magic'
    f_number = 10
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 11)]
    label = "att11"
    data[label] = choosen_data[label]

elif(dataset_name=="skin_nonSkin"):
    print(dataset_name)
    data_path = r'..\Data\skin_NonSkin\skin_NoSkin.csv'
    choosen_data = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,5)])
    #choosen_data = choosen_data[:100]
    X_names_ds = ["att" + str(i) for i in range(1, 4)]
    y_names = "att4"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'skin_nonSkin'
    f_number = 3
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 4)]
    label = "att4"
    data[label] = choosen_data[label]

elif(dataset_name=="cars"):
    print(dataset_name)
    data_path = r'..\Data\car\car.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 8)])
    X_names_ds = ["att" + str(i) for i in range(1, 7)]
    y_names = "att7"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'cars'
    f_number = 6
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 7)]
    label = "att7"
    data[label] = choosen_data[label]

elif(dataset_name=="abalone"):
    print(dataset_name)
    data_path = r'..\Data\abalone\abalone.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 10)])
    X_names_ds = ["att" + str(i) for i in range(1, 9)]
    y_names = "att9"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'abalone'
    f_number = 8
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 9)]
    label = "att9"
    data[label] = choosen_data[label]

elif(dataset_name=="bank"):
    print(dataset_name)
    data_path = r'..\Data\bank\bank.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 18)])
    X_names_ds = ["att" + str(i) for i in range(1, 17)]
    y_names = "att17"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'bank'
    f_number = 16
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 17)]
    label = "att17"
    data[label] = choosen_data[label]

elif(dataset_name=="wine"):
    print(dataset_name)
    data_path = r'..\Data\wine\winequality-white.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 13)])
    X_names_ds = ["att" + str(i) for i in range(1, 12)]
    y_names = "att12"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'wine'
    f_number = 11
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 12)]
    label = "att12"
    data[label] = choosen_data[label]

elif(dataset_name=="blood"):
    print(dataset_name)
    data_path = r'..\Data\blood_transfusion\blood.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 6)])
    X_names_ds = ["att" + str(i) for i in range(1, 5)]
    y_names = "att5"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'blood'
    f_number = 4
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 5)]
    label = "att5"
    data[label] = choosen_data[label]

elif(dataset_name=="wifi"):
    print(dataset_name)
    data_path = r'..\Data\wifi\wifi.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 9)])
    X_names_ds = ["att" + str(i) for i in range(1, 8)]
    y_names = "att8"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'wifi'
    f_number = 7
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 8)]
    label = "att8"
    data[label] = choosen_data[label]

elif(dataset_name=="chess"):
    print(dataset_name)
    data_path = r'..\Data\chess\chess.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 8)])
    X_names_ds = ["att" + str(i) for i in range(1, 7)]
    y_names = "att7"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'chess'
    f_number = 6
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 7)]
    label = "att7"
    data[label] = choosen_data[label]

elif(dataset_name=="letters"):
    print(dataset_name)
    data_path = r'..\Data\letter_recognition\letter_recognition.csv'
    choosen_data = pd.read_csv(data_path, names=["att" + str(i) for i in range(1, 18)])
    X_names_ds = ["att" + str(i) for i in range(1, 17)]
    y_names = "att17"
    encode_categorial(choosen_data)
    un_class = len(choosen_data[y_names].unique())
    print(un_class)
    db_name = 'letters'
    f_number = 16
    data = choosen_data[X_names_ds].copy()
    features_unary = []
    features_binary = ["att" + str(i) for i in range(1, 17)]
    label = "att17"
    data[label] = choosen_data[label]

else:
    exit(0)



for delete_used_f in [True,False]:
    for criterion_name, criterion_function in all_criterions.items():
        for cur_depth in [None,5,6,10,15]:
            base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
                auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,cur_depth,result_path,criterion_function,delete_used_f)

            dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])
            write_to_excel()












