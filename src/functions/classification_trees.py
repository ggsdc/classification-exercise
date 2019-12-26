import itertools as it
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


def decision_trees(X_train, y_train, X_test, y_test, folds, n_attributes):
    results = dict()
    criterions = ['gini', 'entropy']
    splitters = ['random', 'best']
    fold = 1
    for c, s in it.product(criterions, splitters):
        results[(c,s)] = dict()
        for idx_train, idx_test in folds.split(X_train, y_train):
            results[(c,s)][fold] = 0
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]

            model = DecisionTreeClassifier(criterion=c, splitter=s)
            model.fit(X_train_folds, y_train_folds)
            results[(c,s)][fold] = model.score(X_test_folds, y_test_folds)
            print((c, s), fold, (results[(c, s)][fold]))
            fold += 1

    best_acc = 0
    best_config = ''
    for i in results:
        total_acc = 0
        for j in results[i]:
            total_acc += results[i][j]
        mean_acc = total_acc / 10
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_config = i

    print(best_config, best_acc)

    best_model = DecisionTreeClassifier(criterion = best_config[0], splitter=best_config[1])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    final_results = dict()
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(X_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr


    return final_results