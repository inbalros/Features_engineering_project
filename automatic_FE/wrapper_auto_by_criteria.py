
#############################
#                           #
#  data sets - to evaluate  #
#                           #
#############################
import numpy as np
import pandas as pd
from automatic_FE.auto_by_criteria import *


dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'Accuracy_base', 'Criteria_base','number_of_folds','number_of_trees_per_fold','number_of_rounds', 'added_features',
         'accuracy_after', 'criteria_after'])

writerResults = pd.ExcelWriter(r"D:\GitHub\Features_engineering_project\results\results_1.xlsx")

def write_to_excel():
    dfAllPred.to_excel(writerResults,'Cluster_Decomposition')
    writerResults.save()

def criteria_number_of_leafs(tree):
    return tree.tree_.n_leaves




print("start - letters")
#1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition.csv'
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names = ["att"+str(i) for i in range(1,4)]
y_names = "att17"
db_name='letters'
f_number=17
encode_categorial(letters)
un_class = len(letters[y_names].unique())
data = letters[X_names]
features_unary = []
features_binary = ["att"+str(i) for i in range(1,4)]
label = "att17"
data[label] = letters[label]
depth = 5  ###################################### add as parameter

result_path =r"D:\GitHub\Features_engineering_project\results\results.txt"

number_of_kFolds = 5
number_of_trees_per_fold = 5

criterion_name='number_of_leaves'
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names,y_names,features_unary,features_binary,depth,result_path,criteria_number_of_leafs)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()
#(np.array(list((acu_test)),central_clf))