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
from Utility import plots

# ======================================================================================================================
# Data Pre-processing

# data_train, data_val, data_test = data_preprocessing(args...)

# prep.save_data(util.celeba_set, 'landmarks.npy', 'genders.npy')
# prep.save_data(util.celeba_test_set, 'landmarks_test.npy', 'genders_test.npy')

original_landmarks, genders = prep.load_data('landmarks.npy', 'genders.npy')
landmarks_test, genders_test = prep.load_data('landmarks_test.npy', 'genders_test.npy')

landmarks, feat_num = prep.split_and_label_features(original_landmarks, [])
landmarks_test, __ = prep.split_and_label_features(landmarks_test, [])

# Splitting data into train/test/val
train_ratio = 0.80
validation_ratio = 0.10
#test_ratio = 0.10

# Doing the split into train and test, with shuffle
tr_X, te_X, tr_Y, te_Y = train_test_split(landmarks, genders, test_size=1-train_ratio, random_state=42)

print(len(tr_X))
print(tr_X.shape)

# Reshaping the features into 2 dimensions based on data length and number of features
tr_X = tr_X.reshape(len(tr_X), feat_num)
tr_Y = list(tr_Y)

te_X = te_X.reshape(len(te_X), feat_num)
te_Y = list(te_Y)

#======= PANDAS STYLE =======#
# Getting dataframe of all images, with 136 columns corresponding to each feature
X_train = pd.DataFrame(data=tr_X)
Y_train = pd.DataFrame(data=tr_Y)

X_test = pd.DataFrame(data=te_X)
Y_test = pd.DataFrame(data=te_Y)

celeba_test_set = pd.DataFrame(data=landmarks_test)
genders_test = pd.DataFrame(data=genders_test)

Y_train = Y_train[0]
Y_test= Y_test[0]

# THIS SCALER REDUCES ACCURACY
# scaler = preprocessing.MinMaxScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#====== To be moved =====
def rfecv_tool(estimator):
  # The "accuracy" scoring is proportional to the number of correct
  # classifications
  rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5),
                scoring='accuracy')
  rfecv.fit(X_train, Y_train)
  print('score', rfecv.score(X_test, Y_test))
  print("Optimal number of features : %d" % rfecv.n_features_)

  # Plot number of features VS. cross-validation scores
  plt.figure()
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score (nb of correct classifications)")
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
  plt.show()
  return rfecv, X_train.columns[rfecv.support_]

# Create the RFE object and compute a cross-validated score.
lr = LogisticRegression(solver='lbfgs', multi_class='auto')
lr_rfecv, lr_rfecv_features = rfecv_tool(lr)
print('selected features', lr_rfecv_features)
print(models.test_models(X_train[lr_rfecv_features], Y_train, X_test[lr_rfecv_features], Y_test))


# fs = SelectKBest(chi2, k='all')
# fs.fit_transform(X_train, Y_train)
# scores = pd.Series(fs.scores_, index = X_train.columns)
# scores = scores.sort_values()
# print(scores)
#
# features = list(scores[scores > 1].keys())
# print(features)

# apply feature selection

#print(models.test_models(X_train[features], Y_train, X_test[features], Y_test))

# print("Testing models...")
#
# model_tests = models.test_models(models.models, tr_X, tr_Y, te_X, te_Y)
#
# print(model_tests['Name'])
# print(model_tests['Score'])

#print(training_features)

# normalizing tr_X and te_X with preprocessing.normalize(tr_X) reduced accuracy across the board

# Model Validation Stuff

param_grid = {
    'C': [0.5, 1, 10, 15]}
    #'kernel' : ('linear', 'poly'),
    #'degree': [1, 2, 3, 4, 5]}

#plots.plot_validation_curve(SVC(kernel='poly', degree=4), tr_X, tr_Y, "C", [0.001, 0.01, 0.1, 1])

# grid = GridSearchCV(SVC(), param_grid, cv=5)
# grid.fit(tr_X, tr_Y)
# print(grid.best_params_)

# ======================================================================================================================
# Task A1
# test all basic models on data with all features to get preliminary accuracies
# pick best performing model
# do feature selection - CV on number of features? # need to shuffle training data
# hyper paramter tuning - tune parameters in model with CV
    # - plot training score and cross validation score lines for diff hyper parameters
# select best performing model and give test score

# find best C value... validation curve not work
model_A1 = A1(c=0.1, kernel='poly', degree=4)                  # Build model object.

print("Training Model...")
acc_A1_train = model_A1.train(X_train[lr_rfecv_features], Y_train, X_test[lr_rfecv_features], Y_test)
#acc_A1_train = model_A1.train(tr_X, tr_Y, te_X, te_Y)  # Train model based on the training set (you should fine-tune your model based on validation set.)

print(acc_A1_train)

cross_validated_training_acc = model_A1.cross_validate(tr_X, tr_Y, 5)

print(np.mean(cross_validated_training_acc)) # If i use this sklearn cross validation technique, i dont need to split for a validation set

print("Testing Model on celeba_set_test...")
acc_A1_test = model_A1.test(celeba_test_set[lr_rfecv_features], genders_test)    # Test model based on the test set.

print(acc_A1_test)

#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
# feature selection probably important here

model_A2 = A2()
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()
#Clean up memory/GPU etc...


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
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))
