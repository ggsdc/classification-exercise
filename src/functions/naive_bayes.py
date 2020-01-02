from sklearn.naive_bayes import GaussianNB

def naive_bayes(X_train, y_train, X_test, y_test, folds, n_attributes):
    results = dict()
    fold = 1
    for idx_train, idx_test in folds.split(X_train, y_train):
        results[fold] = 0
        X_train_folds = X_train[idx_train]
        X_test_folds = X_train[idx_test]
        y_train_folds = y_train[idx_train]
        y_test_folds = y_train[idx_test]

        model = GaussianNB()
        model.fit(X_train_folds, y_train_folds)
        results[fold] = model.score(X_test_folds, y_test_folds)
        print(fold, (results[fold]))
        fold += 1

    return results