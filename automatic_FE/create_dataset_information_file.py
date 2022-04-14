# import required module
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from STree_ObliqueTreeBasedSVM import *


# assign directory
directory = r'C:\Users\User\Documents\GitHub\Features_engineering_project\New_DB'


def encode_categorical(df,label):
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

def ger_prediction(train,x_names,y_names):
    central_clf = RandomForestClassifier(n_estimators=100, random_state=None).fit(train[x_names], np.array(train[y_names]))

    prediction = central_clf.predict(train[x_names])

    central_clf = XGBClassifier(max_depth=100000, n_estimators=100).fit(train[x_names],np.array(train[y_names]))

    prediction = central_clf.predict(train[x_names])
    central_clf = Stree(max_depth=100000).fit(train[x_names], np.array(train[y_names]))

    prediction = central_clf.predict(train[x_names])

l = []
# iterate over files in
# that directory
#file = open(r'C:\Users\User\Documents\GitHub\Features_engineering_project\config\config_data.txt', 'w')
with open(r'C:\Users\User\Documents\GitHub\Features_engineering_project\config\config_data.txt', 'w') as file:

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            df= pd.read_csv(f,index_col=False)
            #print(f)
            rows= len(df)
            cols = len(df.columns)

            if rows < 20000 and cols<16:
                l.append(f)

                label = df.columns[-1]
                df = df.replace(['?'], None)

                try_data = df.copy().select_dtypes(include='object')
                category_columns = try_data.columns

                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col].fillna(mode(df[col].astype(str)).mode[0], inplace=True)
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)

                allN = f.split("\\")
                name = allN[-1]
                db_name = name.split(".")[0]
                file.write(db_name+",New_DB//"+name+","+str(cols-1)+"\n")
                #encode_categorical(df,category_columns,label)
                #ger_prediction(df, df.columns[0:-1], df.columns[-1])




print(len(l))
print(l)