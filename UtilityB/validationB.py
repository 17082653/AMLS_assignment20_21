
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression,  SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve


def feature_selectionB(classifier, X_train, Y_train, X_test, Y_test, score_func=[chi2,f_classif], threshold=[0,0.5,1,1.5,2,2.5,2.8]):
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

            test_acc = classifier.train(X_train[features], Y_train, X_test[features], Y_test)

            if test_acc > max_acc:
                max_acc = test_acc
                best_features = features

            print(test_acc)

    return best_features