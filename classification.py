from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy
import warnings


def sparse_tocoo(temp_y_labels):
    y_labels = [[] for x in range(temp_y_labels.shape[0])]
    cy = temp_y_labels.tocoo()
    for i, j in zip(cy.row, cy.col):
        y_labels[i].append(j)
    assert sum(len(l) for l in y_labels) == temp_y_labels.nnz
    return y_labels


def get_dataset_for_classification(X, y, train_percent):
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=1 - train_percent)
    y_train = sparse_tocoo(y_train_)
    y_test = sparse_tocoo(y_test_)
    return X_train, X_test, y_train_, y_train, y_test_, y_test


def predict_top_k(classifier, X, top_k_list):
    assert X.shape[0] == len(top_k_list)
    probs = numpy.asarray(classifier.predict_proba(X))
    all_labels = []
    for i, k in enumerate(top_k_list):
        probs_ = probs[i, :]
        try:
            labels = classifier.classes_[probs_.argsort()[-k:]].tolist()
        except AttributeError:  # for eigenpro
            labels = probs_.argsort()[-k:].tolist()
        all_labels.append(labels)
    return all_labels


def get_classifer_performace(classifer, X_test, y_test, multi_label_binarizer):
    top_k_list_test = [int(sum(l)) for l in y_test]
    y_test_pred = predict_top_k(classifer, X_test, top_k_list_test)

    # y_test_transformed = multi_label_binarizer.transform(y_test)

    y_test_transformed = y_test
    y_test_pred_transformed = numpy.zeros((y_test.shape[0], y_test.shape[1]))
    # y_test_pred_transformed = multi_label_binarizer.transform(y_test_pred)

    for i in range(len(y_test_pred)):
        for ele in y_test_pred[i]:
            y_test_pred_transformed[i][ele] = 1.0

    results = {}
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_test_transformed, y_test_pred_transformed, average=average)
    results["accuracy"] = accuracy_score(y_test_transformed, y_test_pred_transformed)

    # print ("======================================================")
    # print("Best Scores with best params: {}".format(str(classifer.get_params()["estimator"]).replace("\n", "")))
    # for metric_score in results:
    #     print (metric_score, ": ", results[metric_score])
    # print ("======================================================")

    return results


def logistic_regression_classification(X_train, X_test, y_train, y_test, multi_label_binarizer, grid_search=True):
    warnings.filterwarnings('always')
    lf_classifer = OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=1)
    if grid_search:
        parameters = {
            "estimator__penalty": ["l1", "l2"],
            "estimator__C": [0.1, 1, 10]  # 0.001, 0.01 100
        }

        lf_classifer = GridSearchCV(lf_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=1, verbose=0)

    lf_classifer.fit(X_train, y_train)
    results = get_classifer_performace(lf_classifer, X_test, y_test, multi_label_binarizer)

    return lf_classifer, results
