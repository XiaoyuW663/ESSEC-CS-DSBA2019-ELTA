# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:52:55 2020

@author: wong
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#%%run model and print test accuracy &f1-score
def run_model(clf,X_train,y_train,X_test,y_test):
    """
    @param:clf - a sklearn classifier e.g. RandomForestClassifier()
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1
    
      
#%%
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

classifiers = {
    'DecisionTree':tree(splitter='best', min_samples_split=10, min_samples_leaf= 6, max_features=100, 
                        max_depth= 30, criterion= 'gini'),
                        
    'Bagging':BaggingClassifier(n_estimators= 300, max_samples =0.7999999999999999,max_features=40),
    'RandomForest':RandomForestClassifier(n_estimators= 230, min_samples_split= 2, min_samples_leaf=2, 
                                          max_features=70, max_depth=45),
    
    'AdaBoost':AdaBoostClassifier(n_estimators=260, learning_rate=0.001, 
                                  base_estimator=tree(class_weight=None, criterion='gini', max_depth=10,
                                                      max_features=30, max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                                      min_samples_leaf=5, min_samples_split=5,
                                                      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                                      splitter='best')),
                                                                                             
    'GradientBoostingTree':GradientBoostingClassifier(n_estimators=175,min_samples_split=2,
                                                      min_samples_leaf=12, max_features=40,
                                                      max_depth=8, learning_rate= 0.1),
    'XGBoost':XGBClassifier(subsample=0.65, reg_lambda =1.0, objective='multi:softmax', n_estimators= 190,
                            min_child_weight= 0.6,max_depth= 38,learning_rate=0.1, gamma= 0.05, 
                            colsample_bytree=0.9, colsample_bylevel=0.5)
 }
