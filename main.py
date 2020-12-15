import os


import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

from Utility import utility as util
from Utility import models
from Utility import plots

# ======================================================================================================================
# Data preprocessing

# data_train, data_val, data_test = data_preprocessing(args...)

# landmark_features, gender_labels = util.extract_features_labels(util.celeba_set)
# np.save('landmarks.npy', landmark_features)
# np.save('genders.npy', gender_labels)

# landmark_features, gender_labels = util.extract_features_labels(util.celeba_test_set)
# np.save('landmarks_test.npy', landmark_features)
# np.save('genders_test.npy', gender_labels)

genders = np.load('genders.npy')
landmarks = np.load('landmarks.npy')

genders_test = np.load('genders_test.npy')
landmarks_test = np.load('landmarks_test.npy')

# Splitting data into training and test
# train/test/val
train_ratio = 0.80
#validation_ratio = 0.10
#test_ratio = 0.10

# Doing the split into train and test, with shuffle
tr_X, te_X, tr_Y, te_Y = train_test_split(landmarks, genders, test_size=1-train_ratio, random_state=42)

print(len(tr_X))

# Splitting validation set off off train set
#val_X, te_X, val_Y, te_Y = train_test_split(te_X, te_Y, test_size=test_ratio/(test_ratio+validation_ratio), shuffle=False)

#print(len(te_X))
#print(len(val_X))

# Reshaping the features into 2 dimensions
tr_X = tr_X.reshape(len(tr_X), 68*2)
tr_Y = list(tr_Y)

te_X = te_X.reshape(len(te_X), 68*2)
te_Y = list(te_Y)

# val_X = te_X.reshape(len(val_X), 68*2)
# val_Y = list(val_Y)

landmarks_test = landmarks_test.reshape(len(landmarks_test), 68*2)
genders_test = list(genders_test)

# normalizing tr_X and te_X with preprocessing.normalize(tr_X) reduced accuracy across the board

# print("Testing models...")
#
# model_tests = models.test_models(models.models, tr_X, tr_Y, te_X, te_Y)
#
# print(model_tests['Name'])
# print(model_tests['Score'])

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
model_A1 = A1(c=1, kernel='poly', degree=4)                  # Build model object.


print("Training Model...")
acc_A1_train = model_A1.train(tr_X, tr_Y, te_X, te_Y)  # Train model based on the training set (you should fine-tune your model based on validation set.)

print(acc_A1_train)

cross_validated_training_acc = model_A1.cross_validate(tr_X, tr_Y, 5)

print(np.mean(cross_validated_training_acc)) # If i use this sklearn cross validation technique, i dont need to split for a validation set

print("Testing Model on celeba_set_test...")
acc_A1_test = model_A1.test(landmarks_test, genders_test)    # Test model based on the test set.

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
