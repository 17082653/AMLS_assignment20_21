# ======================================================================================================================
# Task B2 model class.
# ======================================================================================================================
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class B2:

    # The classifier object.
    def __init__(self):
        self.classifier = KNeighborsClassifier()

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