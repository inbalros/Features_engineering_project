
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance



print("start - magic04")
#2) data set = magic04 https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
data_path = r'..\\Data\\magic04\\magic.csv'
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
    return (sorted(tup, key=lambda x: x[0],reverse=True ))

import matplotlib.pyplot as plt

def create_permutation_importance(rf,X_test,y_test):
    result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()[::-1][:len(result.importances_mean)]

    permutation_importance_list = list(((imp, name)) for name, imp in zip(X_test.columns[sorted_idx], result.importances_mean[sorted_idx].T))

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()
    print(permutation_importance_list)
    return permutation_importance_list

def create_impurity_based_importance(rf,X_names):
    imp_impurity_list = list(((imp, name)) for name, imp in zip(X_names, rf.feature_importances_))
    new_imp_impurity_list = sorted(imp_impurity_list, key=lambda x: x[0],reverse=True )
    print(new_imp_impurity_list)
    return new_imp_impurity_list


kfold = KFold(3)
index = 0
size = data.shape[0]
all_DT_pred = 0
print("gry")
con=0
for train, test in kfold.split(data):
    index += 1

    central_clf = RandomForestClassifier(n_estimators=10, random_state=None) \
    .fit(data.iloc[train][X_names_ds], np.array(data.iloc[train][y_names]))


    create_permutation_importance(central_clf, data.iloc[train][X_names_ds],  np.array(data.iloc[train][y_names]))

    create_impurity_based_importance(central_clf, X_names_ds)


    #prediction = central_clf.predict(data.iloc[test][X_names_ds])  # the predictions labels





    #con+=confusion_matrix(pd.Series.tolist(data.iloc[test][y_names]),prediction)
    #tree_info = (tree.plot_tree(central_clf.estimators_[0]))
    #a = 4

#con = con/index
print(central_clf.feature_importances_)

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
