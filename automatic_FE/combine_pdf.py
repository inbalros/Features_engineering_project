import os
import pandas as pd
cwd = os.path.abspath('')
dir_n = r'C:\Users\User\Documents\phd\שרת\26.12all_db_by_operators'
db_name ='combined'
files = os.listdir(dir_n)
import numpy as np
## Method 1 gets the first sheet of a given file
df = pd.DataFrame()
for file in files:
    if file.endswith('.xlsx'):
        name =os.path.join(dir_n,  file)
        df = df.append(pd.read_excel(name), ignore_index=True)
print(df.head())


df["accuracy_after"] = pd.to_numeric(df["accuracy_after"])
df["accuracy_base"] = pd.to_numeric(df["accuracy_base"])
df["n_leaves_after"] = pd.to_numeric(df["n_leaves_after"])
df["n_leaves_base"] = pd.to_numeric(df["n_leaves_base"])
df["max_depth_after"] = pd.to_numeric(df["max_depth_after"])
df["max_depth_base"] = pd.to_numeric(df["max_depth_base"])
df["node_count_after"] = pd.to_numeric(df["node_count_after"])
df["node_count_base"] = pd.to_numeric(df["node_count_base"])

#df['diff_acc']=(df['accuracy_after']-df['accuracy_base'] if df['added_features']!='[]' else "")
df['diff_acc']= np.where(df['added_features']!= '[]', df['accuracy_after']-df['accuracy_base'], "")
df['diff_n_leaves']= np.where(df['added_features']!= '[]', df['n_leaves_after']-df['n_leaves_base'], "")
df['diff_max_depth']= np.where(df['added_features']!= '[]', df['max_depth_after']-df['max_depth_base'], "")
df['diff_node_count']= np.where(df['added_features']!= '[]', df['node_count_after']-df['node_count_base'], "")

df["diff_acc"] = pd.to_numeric(df["diff_acc"])
df["diff_n_leaves"] = pd.to_numeric(df["diff_n_leaves"])
df["diff_max_depth"] = pd.to_numeric(df["diff_max_depth"])
df["diff_node_count"] = pd.to_numeric(df["diff_node_count"])


df['percentage_acc']= np.where(df['added_features']!= '[]',  (df['diff_acc']/df['accuracy_base'])*100, "")
df['percentage_n_leaves']= np.where(df['added_features']!= '[]', (df['diff_n_leaves']/df['n_leaves_base'])*100, "")
df['percentage_max_depth']= np.where(df['added_features']!= '[]', (df['diff_max_depth']/df['max_depth_base'])*100, "")
df['percentage_node_count']= np.where(df['added_features']!= '[]', (df['diff_node_count']/df['node_count_base'])*100, "")

df['percentage_n_leaves_abs']= df['percentage_n_leaves']*-1
df['percentage_max_depth_abs']=df['percentage_max_depth']*-1
df['percentage_node_count_abs']= df['percentage_node_count']*-1


df["size_model_before"] = pd.to_numeric(df["size_model_before"])
df["size_model_after"] = pd.to_numeric(df["size_model_after"])
df["size_db_train_after"] = pd.to_numeric(df["size_db_train_after"])
df["size_db_train_before"] = pd.to_numeric(df["size_db_train_before"])
df["size_db_test_before"] = pd.to_numeric(df["size_db_test_before"])
df["size_db_test_after"] = pd.to_numeric(df["size_db_test_after"])


df['diff_size_model']= np.where( df['size_model_before']!= '', df['size_model_before']-df['size_model_after'], "")
df['diff_size_db_train']= np.where( df['size_model_before']!= '', df['size_db_train_before']-df['size_db_train_after'], "")
df['diff_size_db_test']= np.where( df['size_model_before']!= '', df['size_db_test_before']-df['size_db_test_after'], "")

'''
df["diff_size_model"] = pd.to_numeric(df["diff_size_model"])
df["diff_size_db_train"] = pd.to_numeric(df["diff_size_db_train"])
df["diff_size_db_test"] = pd.to_numeric(df["diff_size_db_test"])

df['percentage_size_model']= np.where(df['added_features']!= '[]' and df['size_model_before']!= '',  (df['diff_size_model']/df['size_model_before'])*100, "")
df['percentage_size_db_train']= np.where(df['added_features']!= '[]' and df['size_model_before']!= '', (df['diff_size_db_train']/df['size_db_train_before'])*100, "")
df['percentage_size_db_test']= np.where(df['added_features']!= '[]' and df['size_model_before']!= '', (df['diff_size_db_test']/df['size_db_test_before'])*100, "")
'''

name =os.path.join(r'C:\Users\User\Documents\phd\שרת\26.12all_db_by_operators',  'results_'+db_name+'.xlsx')
df.to_excel(name)


'''
## Method 2 gets all sheets of a given file
df_total = pd.DataFrame()
for file in files:                         # loop through Excel files
    if file.endswith('.xlsx'):
        excel_file = pd.ExcelFile(file)
        sheets = excel_file.sheet_names
        for sheet in sheets:               # loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name = sheet)
            df_total = df_total.append(df)
df_total.to_excel('combined_file.xlsx')
'''