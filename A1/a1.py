# ======================================================================================================================
# Task A1 model class. The class A1 essentially acts as a classifier object. The default arguments are that of the
# chosen final model. Certain functions were used in model selection and are mentioned in the report, however are not
# ran in 'main.py' in the final submission. The A1 class ended up being made quite modular and being used across tasks
# ======================================================================================================================
import time
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class A1:

    # The classifier object. Default parameters are not the ones used in the final result
    # The LogisticRegressor with the best parameters found after validation was used. This is created when lr=True is
    # passed to the constructor.
    def __init__(self, c=0.1, kernel='poly', degree=4, lr=False):
        self.c = c
        self.kernel = kernel
        self.degree = degree
        if lr:
            # Best params: {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
            self.classifier = LogisticRegression(penalty='l2', solver='newton-cg', C=0.1, max_iter=4000)
        else:
            # Best params: C=0.1, kernel='poly', degree=4
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