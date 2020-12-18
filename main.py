import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.filterwarnings('ignore')

from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

from UtilityA import pre_processing as prep
from UtilityA import utility as util
from UtilityA import models
from UtilityA import validation
from UtilityA import plots

from UtilityB import pre_processingB as prepB
from UtilityB import utilityB as utilB
from UtilityB import validationB

# ======================================================================================================================
# DATA LOADING/SAVING AND PRE-PROCESSING
# UNSEEN TEST DATA -  CELEBA/GENDERS
landmarks_test, genders_test, __ = prep.data_prep(util.celeba_test_set, 'landmarks_test.npy', 'genders_test.npy')
unseen_test_x, unseen_test_y = prep.convert_to_dataframes(landmarks_test, genders_test)

landmarks_test2, smiles_test, __ = prep.data_prep(util.celeba_test_set, 'landmarks_test2.npy', 'smiles_test.npy')
unseen_test_x2, unseen_test_y2 = prep.convert_to_dataframes(landmarks_test2, smiles_test)

# DATASET LOADING - TASK A
tr_X, te_X, tr_Y, te_Y = prep.split_data_into_sets(util.celeba_set, 'landmarks.npy', 'genders.npy', 0.8, 42)
tr2_X, te2_X, tr2_Y, te2_Y = prep.split_data_into_sets(util.celeba_set, 'landmarks.npy', 'smiles.npy', 0.8, 42, True)

X_train, Y_train = prep.convert_to_dataframes(tr_X, tr_Y)
X_test, Y_test = prep.convert_to_dataframes(te_X, te_Y)

X_2train, Y_2train = prep.convert_to_dataframes(tr2_X, tr2_Y)
X_2test, Y_2test = prep.convert_to_dataframes(te2_X, te2_Y)

# ==================================================================================
# UNSEEN TEST DATA -  CARTOON/EYES/FACES
cartoon_test, cartoon_faces_test, __ = prepB.data_prep(utilB.cartoon_test_set, 'cartoon_landmarks_test.npy', 'cartoon_face_labels_test.npy')
cartoon_test_xB, cartoon_faces_test_yB = prepB.convert_to_dataframes(cartoon_test, cartoon_faces_test)

# Eye extractor for B2 unseen data
if os.path.isfile('eye_feats_test.npy') == False:
    eye_feats_test, eye_labels_test = utilB.extract_eye_features(utilB.cartoon_test_set)
    np.save('eye_feats_test.npy', eye_feats_test)
    np.save('eye_labelsB2_test.npy', eye_labels_test)

eye_feats_test = np.load('eye_feats_test.npy')
eye_labelsB2_test = np.load('eye_labelsB2_test.npy')

XB2_unseen_test, YB2_unseen_test = prepB.convert_to_dataframes(eye_feats_test, eye_labelsB2_test)
# ==================================================================================
# DATASET LOADING - TASK B
tr_BX, te_BX, tr_BY, te_BY = prepB.split_data_into_sets(utilB.cartoon_set, 'landmarksB.npy', 'face_labels.npy', 0.8, 42)

XB_train, YB_train = prepB.convert_to_dataframes(tr_BX, tr_BY)
XB_test, YB_test = prepB.convert_to_dataframes(te_BX, te_BY)

# Extracting eye_features differently
if os.path.isfile('eye_feats.npy') == False:
    eye_features, eye_labels = utilB.extract_eye_features(utilB.cartoon_set)
    np.save('eye_feats.npy', eye_features)
    np.save('eye_labelsB2.npy', eye_labels)

eye_feats = np.load('eye_feats.npy')
eye_labelsB2 = np.load('eye_labelsB2.npy')

XB2_train, YB2_train = prepB.convert_to_dataframes(eye_feats, eye_labelsB2)
tr_B2X, te_B2X, tr_B2Y, te_B2Y=train_test_split(XB2_train, YB2_train, test_size=0.3)

# PARAMETER GRIDS FOR GRID_SEARCH
param_grid_SVC = {
    'C': [0.1, 0.5, 1, 10],
    'kernel' : ['poly'],
    'degree': [2, 3, 4]}

param_grid_lr = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2'],
    'C': [100, 10, 1.0, 0.1, 0.01]}

# ======================================================================================================================

# TASK A1 - Male or Female

# BASIC TEST OF MODELS
# print(models.test_models(X_train, Y_train, X_test, Y_test))

# CLASSIFIER CONSTRUCTION
model_A1 = A1(lr=True)  #  Alternative is SVC:  model_A1 = A1(c=0.1, kernel='poly', degree=4)

# FEATURE SELECTION
print("Performing Task A1 Feature Selection...")
selected_features = validation.feature_selection(X_train, Y_train, X_test, Y_test)
# Unused recursive_feat_elimination...
# selected_features = validation.recursive_feat_elimCV(LogisticRegression(penalty='l2', solver='newton-cg', C=0.1, max_iter=1000), X_train, Y_train, X_test, Y_test)

# TUNING HYPERPARAMETERS - these parameters are already set as default in used models
# best_params = validation.grid_search_CV(LogisticRegression(), param_grid_lr, X_train, Y_train)

print("Training A1 Model...")
acc_A1_train = model_A1.train(X_train[selected_features], Y_train, X_test[selected_features], Y_test)
acc_A1_test = model_A1.test(unseen_test_x[selected_features], unseen_test_y)
print("Cross validated acc: ", np.average(cross_val_score(model_A1.classifier, X_train[selected_features], Y_train, cv=5)))

# ======================================================================================================================

# TASK A2 - Smiling or Not Smiling
model_A2 = A2(lr=True)

# PARAMETER TUNING
# best_params = validation.grid_search_CV(LogisticRegression(), param_grid_lr, X_2train, Y_2train)

# FEATURE SELECTION
print("Performing Task A2 Feature Selection...")
#selected_features_2 = validation.recursive_feat_elimCV(LogisticRegression(penalty='l2', solver='newton-cg', C=0.01, max_iter=1000), X_2train, Y_2train, X_2test, Y_2test)
selected_features_2 = validation.feature_selection(X_2train, Y_2train, X_2test, Y_2test)

print("Training A2 Model...")
acc_A2_train = model_A2.train(X_2train[selected_features_2], Y_2train, X_2test[selected_features_2], Y_2test)
acc_A2_test = model_A2.test(unseen_test_x2[selected_features_2], unseen_test_y2)
print("Cross validated acc: ", np.average(cross_val_score(model_A2.classifier, X_2train[selected_features_2], Y_2train, cv=5)))

# ======================================================================================================================

# Task B1 - Face Shapes
model_B1 = B1(lr=False)
print("Training Task B1 Model...")
acc_B1_train = model_B1.train(XB_train, YB_train, XB_test, YB_test)
acc_B1_test = model_B1.test(cartoon_test_xB, cartoon_faces_test_yB)
print("Cross validated acc: ", np.average(cross_val_score(model_B1.classifier, XB_train, YB_train, cv=5)))

# ======================================================================================================================

# Task B2 - 5 Types of Eye Colors

# BASE MODEL TEST
#print(models.test_models(tr_B2X, tr_B2Y, te_B2X, te_B2Y))

model_B2 = B2()
print("Training Task B2 Model...")
acc_B2_train = model_B2.train(tr_B2X, tr_B2Y, te_B2X, te_B2Y)
acc_B2_test = model_B2.test(XB2_unseen_test, YB2_unseen_test)
print("Cross validated acc: ", np.average(cross_val_score(model_B2.classifier, tr_B2X, tr_B2Y, cv=5)))

# ======================================================================================================================

## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

