import matplotlib as plt
import numpy as np
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, validation_curve

class A1:

    def __init__(self, kernel='linear', degree=3):
        self.kernel = kernel
        self.degree = degree
        self.classifier = svm.SVC(kernel=self.kernel, degree=self.degree)

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

    # def img_SVM(self, training_images, training_labels, test_images, test_labels):
    #     classifier = svm.SVC(kernel='linear')
    #
    #     classifier.fit(training_images, training_labels)
    #
    #     pred = classifier.predict(test_images)
    #
    #     print(pred)
    #
    #     print("Accuracy:", accuracy_score(test_labels, pred))