#=======================================================================================================================
# This file was used for plotting the figures for Task A in the report
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report

# Plots a validation curve for a chosen parameter and parameter value range
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

# Plots classifer confusion matrix
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

# Plot feature selection results, call comes from validation.py
def plot_feat_selection(x, y_dic):
    for model, y_val in y_dic.items():
        if len(y_val) != 0:
            plt.plot(x, y_val, label=model)

    plt.title('Test Accuracy of Models vs Num of Features (f_classif)')
    plt.legend()
    plt.show()

    #
    # # # display scatter plot data
    # # plt.figure(figsize=(10, 8))
    # # plt.title('Scatter Plot', fontsize=20)
    # # plt.xlabel('x', fontsize=15)
    # # plt.ylabel('y', fontsize=15)
    # # plt.scatter(data["x"], data["y"], marker='o')
    #
    # # add labels
    # for label, x, y in zip(data["label"], data["x"], data["y"]):
    #     plt.annotate(label, xy=(x, y))
