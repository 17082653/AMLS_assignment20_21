#=======================================================================================================================
# This file was used for plotting the figures in the report
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report


def plot_validation_curve(model, tr_X, tr_Y, param_name, param_range):
    train_scores, test_scores = validation_curve(
        model, tr_X, tr_Y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve=")
    #plt.xlabel(r"$\gamma$")
    plt.xlabel(param_name.capitalize())
    plt.ylabel("Score")

    # y limit and line-width
    plt.ylim(0.0, 1.1)
    lw = 2

    # plotting
    if param_name == "gamma":
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
    else :
        plt.plot(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")

    plt.show()


def check_classifier(pred, truth):
    s = classification_report(truth, pred, labels=[0, 1, 2])
    print(s)  #  prints classification report
    d = {}  #  gets values like f1, recall and precision from the string
    for l in s.split('\n')[2:5]:
        nums = l.strip().split('      ')
        d[int(nums[0])] = float(nums[2])  # recall

    mat = confusion_matrix(truth, pred)

    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    return np.array(list(d.values()))
