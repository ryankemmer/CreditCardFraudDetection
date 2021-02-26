
#
# ECS Model Test
# Author: Ryan Kemmer
#

import pandas as pd
import dataUtil as du
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, Activation, Input
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def plot_roc(name, fpr, tpr, auc, **kwargs):

    plt.plot(100*fpr, 100*tpr, label= name + ' (area = {:.3f})', linewidth=2, **kwargs).format(auc)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


#Model training

def DNN(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
    ]

    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = METRICS)

    history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
    
    y_pred = model.predict(X_test).ravel()
    fpr, tpr, thresholds_keras = roc_curve(Y_test, y_pred)
    auc = auc(fpr, tpr)
    
    plot_roc('DNN', fpr, tpr, roc, color=colors[3])


def XGBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    xgbTrain = xgb.DMatrix(X_train, label=Y_train)
    xgbTest = xgb.DMatrix(X_test, label=Y_test)
    xgbVal = xgb.DMatrix(X_val, label=Y_val)
    
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'

    num_round = 100
    evallist = [(xgbVal , 'eval'), (xgbTrain, 'train')]

    xgboost = xgb.train(param, xgbTrain, num_round, evallist)
    
    y_pred = xgboost.predict(xgbTest)
    fpr, tpr, thresholds_keras = roc_curve(Y_test, y_pred)
    auc = auc(fpr, tpr)
    
    plot_roc('XGBoost', fpr, tpr, roc, color=colors[2])
    


def CatBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    # Initialize CatBoostClassifier
    cat = CatBoostClassifier(iterations=100,learning_rate=1,depth=2)

    # Fit model
    cat.fit(X_train, Y_train)
    
    y_pred = cat.predict(X_test, prediction_type='RawFormulaVal')
    
    fpr, tpr, thresholds_keras = roc_curve(Y_test, y_pred)
    auc = auc(fpr, tpr)
    
    plot_roc('CatBoost', fpr, tpr, roc, color=colors[1])
    


def RF(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    
    prob = clf.predict_proba(X_test)
    y_pred = []
    
    for c in prob:
        y_pred.append(c[1])
        
    fpr, tpr, thresholds_keras = roc_curve(Y_test, y_pred)
    auc = auc(fpr, tpr)
    
    plot_roc('Random Forest', fpr, tpr, roc, color=colors[0])



#
#Prepare data (ECS)
#

df = pd.read_csv("../ecsAAWeb5App_aiData_wRes.csv")
d = du.data_Util(df, ['PID','EA','EK','FN','BC'])
all_f = [*d.fn_continuous_fields,*d.ea_continuous_fields,*d.ek_continuous_fields, *d.bc_continuous_fields, *d.pid_continuous_fields]
df = df[[*all_f,'BAD_IND']]
df = df.dropna()

#
#Normalize data
#

from sklearn.preprocessing import StandardScaler

for f in all_f:
    df[f] = StandardScaler().fit_transform(df[f].values.reshape(-1,1))
    
#
#Convert data to numpy arrays
#

X = df[all_f].to_numpy()
Y = df['BAD_IND'].to_numpy()


#
#Split data into train, test, and val
#
#
# 70% - train
# 15% - validation
# 15% - test
#
#

X_train, X_split, Y_train, Y_split = train_test_split(X,Y, test_size = 0.3, random_state = 42)
X_test, X_val, Y_test, Y_val = train_test_split(X_split, Y_split, test_size = 0.5, random_state = 42)


#
#Build Models
#

DNN(X_train, Y_train, X_val, Y_val, X_test, Y_test)
XGBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test)
CatBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test)
RF(X_train, Y_train, X_val, Y_val, X_test, Y_test)

plt.legend(loc='lower right')