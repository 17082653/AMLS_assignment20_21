# ======================================================================================================================
# This file contains the code used to validate the models/tune hyperparameters
# ======================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression,  SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve

from Utility import models

#====== To be moved =====
def feature_selection(X_train, Y_train, X_test, Y_test, score_func=[chi2,f_classif], threshold=[0,0.5,1,1.5,2,2.5,2.8]):
    max_acc = 0
    best_features = []
    for func in score_func:
        print("Using score func: ", func)
        for val in threshold:
            # Feature extraction
            model = SelectKBest(score_func=func, k='all')
            model.fit_transform(X_train, Y_train)

            # Summarize scores
            scores = pd.Series(model.scores_, index=X_train.columns)
            scores = scores.sort_values()
            features = list(scores[scores > val].keys())
            filtered = scores[features]
            print('No of features selected: ', len(features), '; Score Threshold: ', val, '; Best Acc: ', max_acc)
            #print(filtered)

            test_acc = models.test_models(X_train[features], Y_train, X_test[features], Y_test)

            if max(test_acc['Score']) > max_acc:
                max_acc = max(test_acc['Score'])
                best_features = features

            print(test_acc)

    return best_features

def recursive_feat_elimCV(model, X_train, Y_train, X_test, Y_test):
    # Create the RFE object and compute a cross-validated score.
    rfecv, rfecv_features = rfecv_tool(model, X_train, Y_train, X_test, Y_test)
    print('selected features', rfecv_features)
    print(models.test_models(X_train[rfecv_features], Y_train, X_test[rfecv_features], Y_test))

    return rfecv_features

def rfecv_tool(estimator, X_train, Y_train, X_test, Y_test):
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

def grid_search_CV(model, param_grid, X_train, Y_train):
    print("Grid Search CV...")
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)

    return grid.best_params_

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



#plots.plot_validation_curve(SVC(kernel='poly', degree=4), tr_X, tr_Y, "C", [0.001, 0.01, 0.1, 1])

