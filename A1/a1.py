import sklearn
import time
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


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

        pred = self.classifier.predict(test_images)


        return accuracy_score(test_labels, pred)

    def test(self, test_images, test_labels):
        pred = self.classifier.predict(test_images)

        return accuracy_score(test_labels, pred)

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