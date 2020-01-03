import datetime as dt

from src.functions.KNN_classifier import knn_classifier
from src.functions.classification_trees import decision_trees
from src.functions.logistic_regression import logistic_regression
from src.functions.metaclassifiers import bagging, random_forest, gradient_boosting
from src.functions.naive_bayes import naive_bayes


def model_evaluation(x_train, y_train, x_test, y_test, folds, n_attributes, skip=False):

    results = dict()
    if not skip:
        t1 = dt.datetime.now()
        results['knn'] = knn_classifier(x_train, y_train, x_test, y_test, folds, n_attributes)
        print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['decision-trees'] = decision_trees(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['naive-bayes'] = naive_bayes(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['logistic-regression'] = logistic_regression(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['bagging'] = bagging(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['random-forest'] = random_forest(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    t1 = dt.datetime.now()
    results['gradient-boosting'] = gradient_boosting(x_train, y_train, x_test, y_test, folds)
    print(dt.datetime.now()-t1)

    # results['rule-induction'] = rule_induction(x_train, y_train, folds)

    return results
