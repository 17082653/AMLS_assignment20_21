import sklearn
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


class A1:

    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.classifier = svm.SVC(kernel=self.kernel)

    def train(self, training_images, training_labels, test_images, test_labels):
        self.classifier.fit(training_images, training_labels)
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