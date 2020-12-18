# ======================================================================================================================
# Task A2 model class. Virtually identical to class A1. Logistic regressor has a different default C value.
# ======================================================================================================================
import time
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class A2:

    def __init__(self, c=1.0, kernel='linear', degree=3, lr=False):
        self.c = c
        self.kernel = kernel
        self.degree = degree
        if lr:
            # best params: {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
            self.classifier = LogisticRegression(penalty='l2', solver='newton-cg', C=0.01, max_iter=1000)
        else:
            self.classifier = svm.SVC(C=self.c, kernel=self.kernel, degree=self.degree)


    def train(self, training_images, training_labels, test_images, test_labels):
        start = time.time()
        self.classifier.fit(training_images, training_labels)
        stop = time.time()

        print(f"Training time: {stop - start:.2f} seconds")

        return self.test(test_images, test_labels)

    def test(self, test_images, test_labels):
        pred = self.classifier.predict(test_images)

        return accuracy_score(test_labels, pred)

    def cross_validate(self, validation_images, validation_labels, cv_folds):
        scores = cross_val_score(self.classifier, validation_images, validation_labels, cv=cv_folds)

        return scores