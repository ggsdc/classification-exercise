from wittgenstein import RIPPER


def rule_induction(x_train, y_train, folds):
    results = dict()
    fold = 1

    for idx_train, idx_test in folds.split(x_train, y_train):
        results[fold] = 0
        x_train_folds = x_train[idx_train]
        x_test_folds = x_train[idx_test]
        y_train_folds = y_train[idx_train]
        y_test_folds = y_train[idx_test]
        print(type(x_train_folds), type(y_train_folds))
        print(x_train_folds, y_train_folds)
        model = RIPPER()
        model.fit(df=x_train_folds, y=y_train_folds)
        results[fold] = model.score(x_test_folds, y_test_folds)
        print(fold, (results[fold]))
        fold += 1

    return True
