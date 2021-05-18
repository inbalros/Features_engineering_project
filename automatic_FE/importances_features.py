
from automatic_FE.auto_by_criteria import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import sys
import os


df_importance_experiment= pd.DataFrame(
    columns=['dataset_name','number_of_all_features','number_of_classes','dataset_size','using_criterion','delete_used_f',
         'feature_importance_base','feature_importance_after',
         'number_of_features','feature_base','feature_after','importance_function',
         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after'])


print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)

dataset_name= str(sys.argv[1])

result_path=os.path.join(one_back,  r"results\results_"+dataset_name+".txt")

index2 = 1
def write_to_excel_df_importance_experiment():
    global index2
    writerResults = pd.ExcelWriter(os.path.join(one_back,  r"results\results_importance_"+dataset_name+"_"+str(index2)+".xlsx"))
    index2+=1
    df_importance_experiment.to_excel(writerResults,'results')
    writerResults.save()

def create_permutation_importance(rf,X_test,y_test):
    result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()[::-1][:len(result.importances_mean)]

    permutation_importance_list = list(((imp, name)) for name, imp in zip(X_test.columns[sorted_idx], result.importances_mean[sorted_idx].T))

    #fig, ax = plt.subplots()
    #ax.boxplot(result.importances[sorted_idx].T,
    #           vert=False, labels=X_test.columns[sorted_idx])
    #ax.set_title("Permutation Importances (test set)")
    #fig.tight_layout()
    #plt.show()
    print(permutation_importance_list)
    return permutation_importance_list

def create_impurity_based_importance(rf,X_names):
    imp_impurity_list = list(((imp, name)) for name, imp in zip(X_names, rf.feature_importances_))
    new_imp_impurity_list = sorted(imp_impurity_list, key=lambda x: x[0],reverse=True )
    print(new_imp_impurity_list)
    return new_imp_impurity_list


def importance_experiment(db_name,f_number,un_class,criterion_name,delete_used_f,ds,last_x_name,all_x_name,y_name,n_estimators,criteria_Function):
    X_train_all_f, X_test_all_f, y_train_all_f, y_test_all_f = train_test_split(ds[all_x_name], ds[y_name], test_size = 0.15, random_state = None)
    X_train_FE_f, X_test_FE_f, y_train_FE_f, y_test_FE_f = train_test_split(ds[last_x_name], ds[y_name], test_size = 0.15, random_state = None)


    rf_all_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
            .fit(X_train_all_f, y_train_all_f)
    rf_FE_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
            .fit(X_train_FE_f, y_train_FE_f)

    permutation_importance_all_f = create_permutation_importance(rf_all_f, X_train_all_f,y_train_all_f)
    impurity_based_importance_all_f = create_impurity_based_importance(rf_all_f, all_x_name)

    permutation_importance_FE_f = create_permutation_importance(rf_FE_f, X_train_FE_f,y_train_FE_f)
    impurity_based_importance_FE_f = create_impurity_based_importance(rf_FE_f, last_x_name)

    min_number_of_f = min(len(last_x_name),len(all_x_name))
    for cur_f_number in range(1,min_number_of_f+1):
        important_f_all_permutation = list(couple[1] for couple in permutation_importance_all_f[:cur_f_number])
        important_f_all_impurity_based = list(couple[1] for couple in impurity_based_importance_all_f[:cur_f_number])
        important_FE_f_permutation = list(couple[1] for couple in permutation_importance_FE_f[:cur_f_number])
        important_FE_f_impurity_based = list(couple[1] for couple in impurity_based_importance_FE_f[:cur_f_number])

        (acu_f_all_permutation, criterion_f_all_permutation, precision_f_all_permutation, recall_f_all_permutation, f_measure_f_all_permutation, roc_f_all_permutation, prc_f_all_permutation,
         n_leaves_f_all_permutation, max_depth_f_all_permutation, node_count_f_all_permutation), clf = \
            predict_kfold(ds, important_f_all_permutation, y_name, criteria_Function, None, 10, 5,None)

        (acu_f_all_impurity_based, criterion_f_all_impurity_based, precision_f_all_impurity_based, recall_f_all_impurity_based,
         f_measure_f_all_impurity_based, roc_f_all_impurity_based, prc_f_all_impurity_based,
         n_leaves_f_all_impurity_based, max_depth_f_all_impurity_based, node_count_f_all_impurity_based), clf = \
            predict_kfold(ds, important_f_all_impurity_based, y_name, criteria_Function, None, 10, 5, None)

        (acu_FE_f_permutation, criterion_FE_f_permutation, precision_FE_f_permutation, recall_FE_f_permutation,
         f_measure_FE_f_permutation, roc_FE_f_permutation, prc_FE_f_permutation,
         n_leaves_FE_f_permutation, max_depth_FE_f_permutation, node_count_FE_f_permutation), clf = \
            predict_kfold(ds, important_FE_f_permutation, y_name, criteria_Function, None, 10, 5, None)

        (acu_FE_f_impurity_based, criterion_FE_f_impurity_based, precision_FE_f_impurity_based, recall_FE_f_impurity_based,
         f_measure_FE_f_impurity_based, roc_FE_f_impurity_based, prc_FE_f_impurity_based,
         n_leaves_FE_f_impurity_based, max_depth_FE_f_impurity_based, node_count_FE_f_impurity_based), clf = \
            predict_kfold(ds, important_FE_f_impurity_based, y_name, criteria_Function, None, 10, 5, None)

        df_importance_experiment.loc[len(df_importance_experiment)] = np.array(
            [db_name, str(f_number), str(un_class), str(ds.shape[0]), criterion_name,str(delete_used_f),
            str(impurity_based_importance_all_f),str(impurity_based_importance_FE_f),str(cur_f_number),
             str(important_f_all_impurity_based),str(important_FE_f_impurity_based),'impurity_based_importance',
             str(acu_f_all_impurity_based),str(criterion_f_all_impurity_based),
             str(precision_f_all_impurity_based), str(recall_f_all_impurity_based), str(f_measure_f_all_impurity_based), str(roc_f_all_impurity_based), str(prc_f_all_impurity_based),
             str(n_leaves_f_all_impurity_based), str(max_depth_f_all_impurity_based), str(node_count_f_all_impurity_based),
             str(acu_FE_f_impurity_based), str(criterion_FE_f_impurity_based),
             str(precision_FE_f_impurity_based), str(recall_FE_f_impurity_based), str(f_measure_FE_f_impurity_based),
             str(roc_FE_f_impurity_based), str(prc_FE_f_impurity_based), str(n_leaves_FE_f_impurity_based),
             str(max_depth_FE_f_impurity_based), str(node_count_FE_f_impurity_based)])

        df_importance_experiment.loc[len(df_importance_experiment)] = np.array(
            [db_name, str(f_number), str(un_class), str(ds.shape[0]), criterion_name, str(delete_used_f),
             str(permutation_importance_all_f), str(permutation_importance_FE_f), str(cur_f_number),
             str(important_f_all_permutation), str(important_FE_f_permutation), 'permutation_importance',
             str(acu_f_all_permutation), str(criterion_f_all_permutation),
             str(precision_f_all_permutation), str(recall_f_all_permutation), str(f_measure_f_all_permutation),
             str(roc_f_all_permutation), str(prc_f_all_permutation),
             str(n_leaves_f_all_permutation), str(max_depth_f_all_permutation), str(node_count_f_all_permutation),
             str(acu_FE_f_permutation), str(criterion_FE_f_permutation),
             str(precision_FE_f_permutation), str(recall_FE_f_permutation), str(f_measure_FE_f_permutation),
             str(roc_FE_f_permutation), str(prc_FE_f_permutation), str(n_leaves_FE_f_permutation),
             str(max_depth_FE_f_permutation), str(node_count_FE_f_permutation)])

        write_to_excel_df_importance_experiment()

