from sklearn.naive_bayes import GaussianNB


def naive_bayes(x_train, y_train, folds):
    results = dict()
    fold = 1
    for idx_train, idx_test in folds.split(x_train, y_train):
        results[fold] = 0
        x_train_folds = x_train[idx_train]
        x_test_folds = x_train[idx_test]
        y_train_folds = y_train[idx_train]
        y_test_folds = y_train[idx_test]

        model = GaussianNB()
        model.fit(x_train_folds, y_train_folds)
        results[fold] = model.score(x_test_folds, y_test_folds)
        print(fold, (results[fold]))
        fold += 1

    return results
