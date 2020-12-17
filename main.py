import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression,  SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

from Utility import pre_processing as prep
from Utility import utility as util
from Utility import models
from Utility import validation
from Utility import plots

# ======================================================================================================================
# Data Pre-processing
# UNSEEN TEST DATA -  CELEBA/GENDERS
landmarks_test, genders_test, __ = prep.data_prep(util.celeba_test_set, 'landmarks_test.npy', 'genders_test.npy')
unseen_test_x, unseen_test_y = prep.convert_to_dataframes(landmarks_test, genders_test)

# DATASET LOADING
tr_X, te_X, tr_Y, te_Y = prep.split_data_into_sets(util.celeba_set, 'landmarks.npy', 'genders.npy', 0.8, 42)
tr2_X, te2_X, tr2_Y, te2_Y = prep.split_data_into_sets(util.celeba_set, 'landmarks.npy', 'smiles.npy', 0.8, 42, True)

X_train, Y_train = prep.convert_to_dataframes(tr_X, tr_Y)
X_test, Y_test = prep.convert_to_dataframes(te_X, te_Y)

X_2train, Y_2train = prep.convert_to_dataframes(tr2_X, tr2_Y)
X_2test, Y_2test = prep.convert_to_dataframes(te2_X, te2_Y)

# Destroys accuracy
# scaler = preprocessing.MinMaxScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
#
# scaler = preprocessing.MinMaxScaler()
# X_2train = pd.DataFrame(scaler.fit_transform(X_2train), columns=X_2train.columns)
# X_2test = pd.DataFrame(scaler.transform(X_2test), columns=X_2test.columns)

# ======================================================================================================================
# Task A1

#model_A1 = A1(c=0.1, kernel='poly', degree=4)                  # Build model object.
model_A1 = A1(lr=True)

param_grid_SVC = {
    'C': [0.1, 0.5, 1, 10],
    'kernel' : ['poly'],
    'degree': [2, 3, 4]}

param_grid_lr = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2'],
    'C': [100, 10, 1.0, 0.1, 0.01]}

print(models.test_models(X_train, Y_train, X_test, Y_test))

# best_params = validation.grid_search_CV(LogisticRegression(), param_grid_lr, X_train, Y_train)

selected_features = validation.feature_selection(X_train, Y_train, X_test, Y_test)

# selected_features = validation.recursive_feat_elimCV(LogisticRegression(penalty='l2', solver='newton-cg', C=0.1, max_iter=1000), X_train, Y_train, X_test, Y_test)

print("Training Model...")
acc_A1_train = model_A1.train(X_train[selected_features], Y_train, X_test[selected_features], Y_test)
acc_A1_test = model_A1.test(unseen_test_x[selected_features], unseen_test_y)
print("Task A1 Complete")

print("Training: ", acc_A1_train)
print("Test: ", acc_A1_test)

# ======================================================================================================================
# Task A2
# feature selection probably important here

model_A2 = A2(lr=True)
print("Training Model...")
acc_A2_train = model_A2.train(X_2train, Y_2train, X_2test, Y_2test)
#acc_A2_test = model_A2.test()
print("Task A2 Complete")
print(acc_A2_train)

"""
# ======================================================================================================================
# Task B1
# train my own facial predictor on cartoon set to get features?

model_B1 = B1()
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2

model_B2 = B2()
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()
#Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
acc_B1_test = 'none'
acc_B2_test = 'none'
acc_B1_train = 'none'
acc_B2_test = 'none'
acc_A2_test = 'none'
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
"""
