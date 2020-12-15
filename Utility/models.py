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


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('Linear SVC', LinearSVC()))
models.append(('SVC (Polynomial Kernel)', SVC(kernel='poly')))
models.append(('SVC (RBF kernel)', SVC(kernel='rbf')))
models.append(('Log Reg', LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000)))
models.append(('Gaussian NB', GaussianNB()))
models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
models.append(('Gradient Boosting', GradientBoostingClassifier()))

def test_models(models, X_train, y_train, X_test, y_test):
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

