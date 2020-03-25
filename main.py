# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:53:55 2020

@Group Members:
    (1) CHEN Min
    (2) WANG Xiaoyu
    (3) ZHANG Qihui
'''
"""


import pandas as pd
import numpy as np
import os
from preprocess import sentence2vector
from models import classifiers,run_model

if __name__ == "__main__":
    
    print('********************** Loading data  **********************')
    path = os.getcwd()+'\\data\\'
    X_train = pd.read_csv(path+'X_train.csv', index_col=0)['designation']
    X_test =  pd.read_csv(path+'X_test.csv', index_col=0)['designation']
    Y_train = pd.read_csv(path+'Y_train.csv', index_col=0)['prdtypecode']
    Y_test = pd.read_csv(path+'Y_test.csv', index_col=0)['prdtypecode']

#%%
    print('********************** Preprocessing  **********************')
    method = 'TF-IDF'

    text = list(pd.concat([X_train,X_test],axis = 0,ignore_index = True))
    text2vec = sentence2vector(text,method = method,min_count=3)
    X = text2vec.X

    X_train = X[:len(X_train)]
    X_test = X[len(X_train):]
    y_train = np.array(Y_train)
    y_test = np.array(Y_test)

    #%%
    print('********************** Running models **********************')
    for name,clf in classifiers.items():
        acc, f1 = run_model(clf,X_train,y_train,X_test,y_test)
        print('[Model]: ' + name +' [acc]: ' + str(acc) +' [f1]: '+ str(f1))

