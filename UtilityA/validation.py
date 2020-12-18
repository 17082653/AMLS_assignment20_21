# =====================================================================================================================
# This file contains the code used to validate the models/tune hyperparameters
# ======================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression,  SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve

from UtilityA import models
from UtilityA import plots

# ======================================================================================================================
# Feature selection, based on both chi2 and f_classif score functions. Wide range of score thresholds as well to decide
# on best combination. Runs test on all basic models to decide best score funciton/threshold and features.
# Plotting only works when one score_func is supplied.
def feature_selection(X_train, Y_train, X_test, Y_test, score_func=[chi2, f_classif], threshold=[0,0.5,1,1.5,2,2.5,2.8]):
    feature_selection_results = {'KNN': [], 'SVC': [], 'Linear SVC': [], 'SVC (Polynomial Kernel)': [],
                                 'Log Reg': [], 'Gaussian NB': [], 'Random Forest': [], 'Gradient Boosting': [],
                                 'Average': []}
    num_of_feats = []

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

            # Adding values to plot. Un-commenct for plotting
            #add_feature_selection_results(test_acc, len(features), feature_selection_results, num_of_feats)

            if max(test_acc['Score']) > max_acc:
                max_acc = max(test_acc['Score'])
                best_features = features

            #print(test_acc)

    # Plotting. Un-comment for plotting
    #plots.plot_feat_selection(num_of_feats, feature_selection_results)

    return best_features

# This function adds feature selection results to dictionary and lists for later plotting
def add_feature_selection_results(test_acc, feats, target_y, target_x):
    # appends results
    target_x.append(feats)
    for name in test_acc['Name']:
        target_y.setdefault(name, []).append(test_acc['Score'][test_acc['Name'].index(name)])

# Plots recursive feature elimination using rfecv_tool function
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

# Cross validated grid search that allows us to search for the best parameters
def grid_search_CV(model, param_grid, X_train, Y_train):
    print("Grid Search CV...")
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)

    return grid.best_params_
