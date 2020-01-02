from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def logistic_regression(X_train, y_train, X_test, y_test, folds, n_attributes):
    print("LOGISTIC REGRESSION")
    results = dict()
    fold = 1
    for idx_train, idx_test in folds.split(X_train, y_train):
        results[fold] = 0
        X_train_folds = X_train[idx_train]
        X_test_folds = X_train[idx_test]
        y_train_folds = y_train[idx_train]
        y_test_folds = y_train[idx_test]

        model = LogisticRegression(penalty="none", solver="saga")
        model.fit(X_train_folds, y_train_folds)
        results[fold] = model.score(X_test_folds, y_test_folds)
        print(fold, (results[fold]))
        fold += 1

    best_model = LogisticRegression(penalty="none", solver="saga")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(X_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results