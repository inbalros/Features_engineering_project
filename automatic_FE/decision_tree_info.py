
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn import tree
from automatic_FE.auto_by_criteria import *


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


def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return (sorted(tup, key=lambda x: x[0]))

kfold = KFold(3)
index = 0
size = data.shape[0]
all_DT_pred = 0
print("gry")
for train, test in kfold.split(data):
    index += 1
    central_clf = RandomForestClassifier(n_estimators=10, random_state=None) \
    .fit(data.iloc[train][X_names_ds], np.array(data.iloc[train][y_names]))
    #tree_info = (tree.plot_tree(central_clf.estimators_[0]))
    #a = 4
print(central_clf.feature_importances_)
imp_list = list(((imp,"att"+str(index+1)) for index, imp in enumerate(central_clf.feature_importances_)))
print(imp_list)
print(Sort_Tuple(imp_list))
#central_clf.feature_importances_
#central_clf.estimators_












exit(0)
arr = [15, 2, 4, 8, 9, 5, 10, 23]
n = 8
sum = 23
def f(arr,n,sum):
    for i in range(n):
        curr_sum = arr[i]
        j = i+1
        while j < n:
            if curr_sum == sum:
                print ("Sum found!")
                print("arr["+str(i)+"…"+str(j-1)+"]")
                return
            if curr_sum > sum :
                break
            curr_sum = curr_sum + arr[j]
            j = j + 1
    print ("No sum found")
    print(str(-1))
    return

f(arr,n,sum)

arr = [12, 2, 4, 15, 9, 3, 10, 2]
n = 8
sum = 20

f(arr,n,sum)

arr = [12, 11, 4, 6, 15, 4, 1, 6]
n = 8
sum = 26

f(arr,n,sum)

arr = [3, 8, 5, 2, 12, 7, 1, 6]
n = 8
sum = 35

f(arr,n,sum)
