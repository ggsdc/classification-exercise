from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, BernoulliNB


def naive_bayes(x_train, y_train, x_test, y_test, folds):
    print("NAIVE BAYES")
    results = dict()
    fold = 1
    for idx_train, idx_test in folds.split(x_train, y_train):
        results[fold] = 0
        x_train_folds = x_train[idx_train]
        x_test_folds = x_train[idx_test]
        y_train_folds = y_train[idx_train]
        y_test_folds = y_train[idx_test]

        model = BernoulliNB(binarize=0.1)
        model.fit(x_train_folds, y_train_folds)
        results[fold] = model.score(x_test_folds, y_test_folds)
        print(fold, (results[fold]))
        fold += 1

    best_model = GaussianNB()
    best_model.fit(x_train, y_train)
    y_prediction = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_prediction)
    cr = classification_report(y_test, y_prediction, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(x_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results
