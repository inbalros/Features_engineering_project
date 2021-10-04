#exploratory data analysis to understand our data better and to show you some cool graphs.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from scipy.stats import mode
#from tpot import TPOTClassifier

import os
tr_data = pd.read_csv('../input/saftey_efficay_myopiaTrain.csv')
te_data = pd.read_csv('../input/saftey_efficay_myopiaTest.csv')
print('train shape is: {} \r\ntest shape is: {}'.format(tr_data.shape, te_data.shape))
print(tr_data.dtypes)
try_data = te_data.select_dtypes(include='object')
category_columns = try_data.columns
print(category_columns.shape)
print(tr_data.head())
print(te_data.head())


#So now we see that we have both categorical features and numeric ones - something to think about when we will build our model and do pre process.

#since we now know it's shape we can see that we cant really show it all nicely in one go, so will set the numbers of rows and columns panda shows us and use the transpose function.

#set number of rows and columns to see
pd.options.display.max_rows = 200
pd.options.display.max_columns = 50

#use transposed view of the features
print(tr_data.describe().T)


#Let's check if we have any missing values hiding around there
print(tr_data.isnull().sum())
null_data = tr_data[tr_data.isnull().any(axis=1)]
print(null_data.shape)

#It's seems there is no feature without missing values in it and no row without missing values in at least one feature. so looking ahead we will need to find a way to deal with them. or ignore them. also there are 12450 records with missing label in the train set!
tr_data = tr_data.dropna(subset=['Class'])
print(tr_data.shape)
print('the value counts of the target are:')
print(tr_data.iloc[:,-1].value_counts())
print(tr_data.iloc[:,-1].value_counts().plot(kind = 'bar'))
#As we can see the data is unbalanced in the target class.

for i,feat in enumerate(tr_data.columns[:-1]): #we start from the second feature as the first one is the item id
    print('the value counts of feature {} are:'.format(feat))
    print(tr_data[feat].value_counts())
    # to show the entire features list coment out the following line
    if i > 8: break


## this function lets us see the features in a bar graph to view how are they distributed.
def value_counts_bars(dat,rows = 5, cols = 5):
    _,ax = plt.subplots(rows,cols,sharey='row',sharex='col',figsize = (cols*5,rows*5))
    for i,feat in enumerate(dat.columns[:(rows*cols)]):
        dat[feat].value_counts().iloc[:20].plot(kind = 'bar',ax=ax[int(i/cols), int(i%cols)],title='value_counts {}'.format(feat))

value_counts_bars(tr_data.iloc[:,:],8,8)
# value_counts_plots(tr_data.iloc[:,25:],6,6)

correlation = tr_data.iloc[:,:-1].corr()
# to get a better sense of the correlation between features we can use a heatmap
plt.figure(figsize=(17,12))
sns.heatmap(correlation,cmap='RdBu')

#we would like to see the correlation with the target variable
f, ax = plt.subplots(figsize=(20,17))
target_corr = tr_data.iloc[:,:].corr()
sns.heatmap(target_corr,vmax=0.8,square=True,cmap='RdBu')

#Lets look at the corrleation with the target in a differnt way.
f, ax = plt.subplots(figsize = (20,5))
target_corr.iloc[:-1,-1].plot(kind = 'bar')

#Pre-process:
for col in tr_data.columns:
    if tr_data[col].dtype == 'object':
        tr_data[col].fillna(mode(tr_data[col].astype(str)).mode[0],inplace=True)
    else:
        tr_data[col].fillna(tr_data[col].mean(),inplace=True)

for col in te_data.columns:
    if te_data[col].dtype == 'object':
        te_data[col].fillna(mode(te_data[col].astype(str)).mode[0],inplace=True)
    else:
        te_data[col].fillna(te_data[col].mean(),inplace=True)

def encode_categorical(data,var_mode):
    le = LabelEncoder()
    for i in var_mode:
        data[i] = le.fit_transform(data[i].astype(str))

print(category_columns)

encode_categorical(tr_data,category_columns)
print(tr_data.head())
encode_categorical(te_data,category_columns)
print(te_data.head())

print(tr_data.columns)


def classification_model(model, data, predicators, target):
    """Generic classification model function with kfold validation"""
    model.fit(data[predicators], data[target])
    preds = model.predict(data[predicators])
    print("Training accuracy is: {}".format(metrics.accuracy_score(preds, data[target])))

    # perform K-fols cross validation
    print(
        "Cross-Validation Score is: {}".format(np.mean(cross_val_score(model, data[predicators], data[target], cv=10))))
    ##fit the model again so that it can be refered outside the function if needed
    model.fit(data[predicators], data[target])

# X_train, X_val, y_train, y_val = train_test_split(tr_data.iloc[:,:-1],tr_data.iloc[:,-1],test_size = 0.2,random_state =12345)
target = 'Class'
pred_vars = tr_data.columns[:-1]
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.2, min_samples_leaf=15, min_samples_split=20, n_estimators=100, subsample=0.6000000000000001)
# model = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.15000000000000002, min_samples_leaf=19, min_samples_split=13, n_estimators=100)
classification_model(model,tr_data,pred_vars,target)

print(te_data.head())

predictions = model.predict_proba(te_data)[:,1]
subm = pd.DataFrame(predictions)
subm.insert( loc=0,column='Id',value=[x+1 for x in range(len(te_data))])
subm.columns = ['Id', 'Class']
subm.to_csv('submit.csv',index=False)
print(subm.head(100))
print(subm.shape)



# tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, scoring='roc_auc')
# tpot.fit(tr_data.iloc[:,:-1], tr_data['Class'])
# tpot.predict(te_data)
# tpot.export('tpot_mnist_pipeline.py')
# %%time

# knn = KNeighborsClassifier(n_jobs=4,n_neighbors=4)
# knn.fit(X_train,y_train)
# knn4_pred = knn.predict(X_val)
# print(confusion_matrix(y_pred=knn4_pred,y_true=y_val))
# sns.heatmap(xticklabels=range(1,10),yticklabels=range(1,10),data = confusion_matrix(y_pred=knn4_pred,y_true=y_val),cmap='Greens')

# dtc = DecisionTreeClassifier(max_depth=100,max_features=92,min_samples_split=2,random_state=12345)
# dtc.fit(X_train,y_train)
# tree_pred = dtc.predict(X_val)
# print(confusion_matrix(y_pred=tree_pred,y_true=y_val))
# sns.heatmap(confusion_matrix(y_pred=tree_pred,y_true=y_val),cmap='Greens',xticklabels=range(1,10),yticklabels=range(1,10))
# print('classification report results:\r\n' + classification_report(y_pred=tree_pred,y_true=y_val))

# dtrain = xgb.DMatrix(data=X_train,label=y_train-1) #xgb classes starts from zero
# dval = xgb.DMatrix(data=X_val,label=y_val-1) #xgb classes starts from zero
# watchlist = [(dval,'eval'), (dtrain,'train')]

# xgb_params = {
#     'eta': 0.05,
#     'max_depth': 7,
#     'subsample': 0.9,
#     'colsample_bytree': 0.9,
#     'colsample_bylevel': 0.7,
#     'alpha':0.1,
#     #'objective': 'binary:logistic',
#     'objective': 'multi:softmax',
#     #'eval_metric': 'auc',
#     'eval_metric': 'mlogloss',
#     'watchlist':watchlist,
#     'print_every_n':5,
#     'min_child_weight':2,
#     'num_class' : 9
# }

# bst = xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=400)
# xgb_pred = bst.predict(dval)
# print(confusion_matrix(y_pred=xgb_pred,y_true=y_val))
# sns.heatmap(confusion_matrix(y_pred=xgb_pred+1,y_true=y_val),cmap='Greens',xticklabels=range(1,10),yticklabels=range(1,10))
# print('classification report results:\r\n' + classification_report(y_pred=xgb_pred+1,y_true=y_val))











