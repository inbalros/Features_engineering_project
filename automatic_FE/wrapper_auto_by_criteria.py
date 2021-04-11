
#############################
#                           #
#  data sets - to evaluate  #
#                           #
#############################
import numpy as np
import pandas as pd
from automatic_FE.auto_by_criteria import *

result_path =r"D:\phd\results\results.txt"


dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'Accuracy_base', 'Criteria_base','number_of_folds','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features',
         'accuracy_after', 'criteria_after'])

index = 21
def write_to_excel():
    global index
    writerResults = pd.ExcelWriter(r"D:\phd\results\results_"+str(index)+".xlsx")
    index+=1
    dfAllPred.to_excel(writerResults,'results')
    writerResults.save()


###############################
#                             #
#      all the criterion      #
#         functions           #
#                             #
###############################

def criteria_number_of_leafs(tree):
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
criterion_name='max_depth'



print("start - magic04")
#2) data set = magic04 https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\magic04\magic_syn.csv'
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\magic04\magic.csv'
magic = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,12)])
X_names_ds = ["att"+str(i) for i in range(1,11)]
y_names = "att11"
encode_categorial(magic)
un_class = len(magic[y_names].unique())
print(un_class)

db_name='magic'
f_number=10
data = magic[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,11)]
label = "att11"
data[label] = magic[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()


print("start - skin_nonSkin")
# 3) data set = skin_NonSkim https://archive.ics.uci.edu/ml/datasets/skin+segmentation#
gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\skin_NonSkin\skin_NoSkin_syn.csv'
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\skin_NonSkin\skin_NoSkin.csv'
skin = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,5)])
X_names_ds = ["att"+str(i) for i in range(1,4)]
y_names = "att4"
encode_categorial(skin)
un_class = len(skin[y_names].unique())
print(un_class)

db_name='skin'
f_number=3
data = skin[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,4)]
label = "att4"
data[label] = skin[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

print("start - cars")
# 4) data set = car https://archive.ics.uci.edu/ml/machine-learning-databases/car/
gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\car\car_syn.csv'
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\car\car.csv'
cars = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
X_names_ds = ["att"+str(i) for i in range(1,7)]
y_names = "att7"
encode_categorial(cars)
un_class = len(cars[y_names].unique())
print(un_class)

db_name='cars'
f_number=6
data = cars[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,7)]
label = "att7"
data[label] = cars[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

print("start - abalone")
#5) data set = abalone : https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names
gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\abalone\abalone_syn.csv'
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\abalone\abalone.csv'
abalone = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,10)])
X_names_ds = ["att"+str(i) for i in range(1,9)]
y_names = "att9"
encode_categorial(abalone)
un_class = len(abalone[y_names].unique())
print(un_class)

db_name='abalone'
f_number=8
data = abalone[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,9)]
label = "att9"
data[label] = abalone[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()



print("start - bank")
# 6) data set = bank  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\bank\bank.csv'
bank = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names_ds = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
encode_categorial(bank)
un_class = len(bank[y_names].unique())
print(un_class)

db_name='bank'
f_number=16
data = bank[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,17)]
label = "att17"
data[label] = bank[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()


print("start - wine")
# 7) data set = wine  https://archive.ics.uci.edu/ml/datasets/Wine+Quality
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\wine\winequality-white.csv'
wine = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,13)])
X_names_ds = ["att"+str(i) for i in range(1,12)]
y_names = "att12"
encode_categorial(wine)
un_class = len(wine[y_names].unique())
print(un_class)

db_name='wine'
f_number=11
data = wine[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,12)]
label = "att12"
data[label] = wine[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

print("start - blood")
# 8) data set = blood  https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\blood_transfusion\blood.csv'
blood = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,6)])
X_names_ds = ["att"+str(i) for i in range(1,5)]
y_names = "att5"
encode_categorial(blood)
un_class = len(blood[y_names].unique())
print(un_class)

db_name='blood'
f_number=4
data = blood[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,5)]
label = "att5"
data[label] = blood[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

print("start - wifi")
# 9) data set = wifi  https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\wifi\wifi.csv'
wifi = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,9)])
X_names_ds = ["att"+str(i) for i in range(1,8)]
y_names = "att8"
encode_categorial(wifi)
un_class = len(wifi[y_names].unique())
print(un_class)

db_name='wifi'
f_number=7
data = wifi[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,8)]
label = "att8"
data[label] = wifi[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()



print("start - chess")
# 10) data set = chess https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\chess\chess.csv'
chess = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
X_names_ds = ["att"+str(i) for i in range(1,7)]
y_names = "att7"
encode_categorial(chess)
un_class = len(chess[y_names].unique())
print(un_class)

db_name='chess'
f_number=6
data = chess[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,7)]
label = "att7"
data[label] = chess[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()




print("start - letters")
#1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition.csv'
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names_ds = ["att"+str(i) for i in range(1,17)]
y_names = "att17"
encode_categorial(letters)
un_class = len(letters[y_names].unique())
print(un_class)

db_name='letters'
f_number=16
data = letters[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,17)]
label = "att17"
data[label] = letters[label]

delete_used_f=True
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()

delete_used_f=False
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_max_depth,delete_used_f)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(delete_used_f),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()


exit(0)






dfAllPred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'Accuracy_base', 'Criteria_base','number_of_folds','number_of_trees_per_fold','number_of_rounds', 'added_features',
         'accuracy_after', 'criteria_after'])

writerResults = pd.ExcelWriter(r"D:\phd\results\results_1.xlsx")

def write_to_excel():
    dfAllPred.to_excel(writerResults,'results')
    writerResults.save()

def criteria_number_of_leafs(tree):
    return tree.tree_.n_leaves



print("start - letters")
#1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition.csv'
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names_ds = ["att"+str(i) for i in range(1,4)]
y_names = "att17"
db_name='letters'
f_number=17
encode_categorial(letters)
un_class = len(letters[y_names].unique())

f_number=17
data = letters[X_names_ds].copy()
features_unary = []
features_binary = ["att"+str(i) for i in range(1,4)]
label = "att17"
data[label] = letters[label]
depth = 5  ###################################### add as parameter
number_of_kFolds = 5
number_of_trees_per_fold = 5


criterion_name='number_of_leaves'
base_acu_test, base_criterion, rounds, added_f_names, acu_test, criterion_cur = \
    auto_F_E(number_of_kFolds,number_of_trees_per_fold,data,X_names_ds,y_names,features_unary,features_binary,depth,result_path,criteria_number_of_leafs)

dfAllPred.loc[len(dfAllPred)] = np.array([db_name,str(f_number),str(un_class),str(data.shape[0]),criterion_name,str(base_acu_test), base_criterion,str(number_of_kFolds),str(number_of_trees_per_fold),str(rounds),str(added_f_names),str(acu_test),str(criterion_cur)])

write_to_excel()
#(np.array(list((acu_test)),central_clf))




exit(0)


