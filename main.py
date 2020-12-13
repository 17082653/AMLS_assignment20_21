import os

import numpy as np
from sklearn.model_selection import train_test_split


from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

from Utility import utility as util

# ======================================================================================================================
# Data preprocessing

#data_train, data_val, data_test = data_preprocessing(args...)

#tr_X, tr_Y, te_X, te_Y = util.get_data(util.celeba_set)

#landmark_features, gender_labels = util.extract_features_labels(util.celeba_set)

# np.save('landmarks.npy', landmark_features)
# np.save('genders.npy', gender_labels)

# ======================================================================================================================
# Task A1

genders = np.load('genders.npy')
landmarks = np.load('landmarks.npy')

tr_X, te_X, tr_Y, te_Y = train_test_split(landmarks, genders, test_size=0.33, random_state=42)

tr_X = tr_X.reshape(len(tr_X), 68*2)
tr_Y = list(tr_Y)

te_X = te_X.reshape(len(te_X), 68*2)
te_Y = list(te_Y)

model_A1 = A1()                  # Build model object.
acc_A1_train = model_A1.train(tr_X, tr_Y, te_X, te_Y)  # Train model based on the training set (you should fine-tune your model based on validation set.)

print(acc_A1_train)

acc_A1_test = model_A1.test()    # Test model based on the test set.
#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2

model_A2 = A2()
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1

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
