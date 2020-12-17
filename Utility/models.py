# ======================================================================================================================
# This file was used to run tests on a variety of models to determine the best performing starting model
# ======================================================================================================================
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_models(X_train, y_train, X_test, y_test):
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVC', SVC()))
    models.append(('Linear SVC', LinearSVC()))
    models.append(('SVC (Polynomial Kernel)', SVC(kernel='poly')))
    models.append(('Log Reg', LogisticRegression()))
    models.append(('My Log Reg', LogisticRegression(penalty='l2', solver='newton-cg', C=0.1, max_iter=1000)))
    models.append(('Gaussian NB', GaussianNB()))
    models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
    models.append(('Gradient Boosting', GradientBoostingClassifier()))

    names = []
    scores = []
    for name, model in models:
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      scores.append(accuracy_score(y_test, y_pred))
      names.append(name)
    average_score = np.average(scores)
    scores.append(average_score)
    names.append('Average')

    tr_split = {'Name': names, 'Score': scores}
    return tr_split

